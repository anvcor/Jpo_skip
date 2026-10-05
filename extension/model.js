// J-PlatPat 验证码识别 —— 纯 JavaScript 推理（不依赖 onnxruntime 等第三方库）
//
// 模型和预处理必须与训练（train.py）完全一致：
//   RGBA → 灰度（PIL 'L' 的整数公式）→ 四边各裁 1 像素黑框 → 缩放到 240×48（复刻 PIL 的 BILINEAR 重采样）
//   → (x/255 − 0.5)/0.5 → 4 个卷积块 → 按行平均 → 全连接 → 4 位 × 10 类
// 权重由 export_web.py 导出到 model/jpp_cnn.bin（float32），结构写在 model/jpp_cnn.json。

const JppModel = (() => {
  let meta = null;
  let weights = null;

  async function load(baseUrl) {
    const [m, buf] = await Promise.all([
      fetch(baseUrl + 'jpp_cnn.json').then((r) => r.json()),
      fetch(baseUrl + 'jpp_cnn.bin').then((r) => r.arrayBuffer()),
    ]);
    if (buf.byteLength !== m.floats * 4) throw new Error(`权重文件大小不对：${buf.byteLength} ≠ ${m.floats * 4}`);
    meta = m;
    weights = new Float32Array(buf);
    return meta;
  }

  // ---------- 预处理：逐像素复刻 PIL ----------

  // PIL convert('L')：L = (R·19595 + G·38470 + B·7471 + 0x8000) >> 16
  function toGray(rgba, w, h) {
    const g = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < g.length; i++, j += 4) {
      g[i] = (rgba[j] * 19595 + rgba[j + 1] * 38470 + rgba[j + 2] * 7471 + 0x8000) >>> 16;
    }
    return g;
  }

  function crop(src, w, h, c) {
    const cw = w - 2 * c, ch = h - 2 * c, out = new Uint8Array(cw * ch);
    for (let y = 0; y < ch; y++) out.set(src.subarray((y + c) * w + c, (y + c) * w + c + cw), y * cw);
    return { data: out, w: cw, h: ch };
  }

  // PIL Resample.c 的 precompute_coeffs + normalize_coeffs_8bpc（BILINEAR，support = 1）
  const PRECISION_BITS = 32 - 8 - 2;
  function coeffs(inSize, outSize) {
    const scale = inSize / outSize;
    const filterscale = Math.max(scale, 1);
    const support = 1.0 * filterscale;
    const ksize = Math.ceil(support) * 2 + 1;
    const bounds = new Int32Array(outSize * 2);
    const kk = new Float64Array(outSize * ksize);
    const ss = 1 / filterscale;
    for (let xx = 0; xx < outSize; xx++) {
      const center = (xx + 0.5) * scale;
      let xmin = Math.trunc(center - support + 0.5);
      if (xmin < 0) xmin = 0;
      let xmax = Math.trunc(center + support + 0.5);
      if (xmax > inSize) xmax = inSize;
      xmax -= xmin;
      let ww = 0;
      for (let x = 0; x < xmax; x++) {
        let t = Math.abs((x + xmin - center + 0.5) * ss);
        const w = t < 1 ? 1 - t : 0;
        kk[xx * ksize + x] = w;
        ww += w;
      }
      for (let x = 0; x < xmax; x++) if (ww !== 0) kk[xx * ksize + x] /= ww;
      bounds[xx * 2] = xmin;
      bounds[xx * 2 + 1] = xmax;
    }
    const ik = new Int32Array(kk.length);
    for (let i = 0; i < kk.length; i++) {
      ik[i] = kk[i] < 0 ? Math.trunc(-0.5 + kk[i] * (1 << PRECISION_BITS)) : Math.trunc(0.5 + kk[i] * (1 << PRECISION_BITS));
    }
    return { ksize, bounds, ik };
  }

  const clip8 = (ss) => {
    const v = Math.floor(ss / (1 << PRECISION_BITS));
    return v < 0 ? 0 : v > 255 ? 255 : v;
  };

  // 先水平后垂直，中间结果取整成 8 位（和 PIL 一样）
  function resizePIL(src, sw, sh, dw, dh) {
    const hc = coeffs(sw, dw), vc = coeffs(sh, dh);
    const tmp = new Uint8Array(dw * sh);
    for (let y = 0; y < sh; y++) {
      for (let xx = 0; xx < dw; xx++) {
        const xmin = hc.bounds[xx * 2], xmax = hc.bounds[xx * 2 + 1], k = xx * hc.ksize;
        let ss = 1 << (PRECISION_BITS - 1);
        for (let x = 0; x < xmax; x++) ss += src[y * sw + x + xmin] * hc.ik[k + x];
        tmp[y * dw + xx] = clip8(ss);
      }
    }
    const out = new Uint8Array(dw * dh);
    for (let yy = 0; yy < dh; yy++) {
      const ymin = vc.bounds[yy * 2], ymax = vc.bounds[yy * 2 + 1], k = yy * vc.ksize;
      for (let xx = 0; xx < dw; xx++) {
        let ss = 1 << (PRECISION_BITS - 1);
        for (let y = 0; y < ymax; y++) ss += tmp[(y + ymin) * dw + xx] * vc.ik[k + y];
        out[yy * dw + xx] = clip8(ss);
      }
    }
    return out;
  }

  // 返回模型输入：灰度 8 位图（训练时 load_dataset 得到的就是这个）
  function preprocess(rgba, w, h) {
    let g = toGray(rgba, w, h), gw = w, gh = h;
    if (meta.crop) ({ data: g, w: gw, h: gh } = crop(g, w, h, meta.crop));
    return resizePIL(g, gw, gh, meta.width, meta.height);
  }

  // ---------- 网络 ----------

  // 3×3 卷积（padding=1）+ ReLU。
  // 先把输入四周补一圈 0，内循环就不用判断边界；每次同时算 4 个输出通道，9 个输入像素读一次用 4 遍。
  function conv3x3Relu(x, C, H, W, L) {
    const O = L.out, plane = H * W, PW = W + 2, pplane = (H + 2) * PW;
    const xp = new Float32Array(C * pplane);
    for (let c = 0; c < C; c++) {
      for (let y = 0; y < H; y++) xp.set(x.subarray(c * plane + y * W, c * plane + (y + 1) * W), c * pplane + (y + 1) * PW + 1);
    }
    const out = new Float32Array(O * plane);
    const w = weights;
    let o = 0;
    for (; o + 4 <= O; o += 4) {
      const q0 = o * plane, q1 = q0 + plane, q2 = q1 + plane, q3 = q2 + plane;
      out.fill(w[L.b + o], q0, q1); out.fill(w[L.b + o + 1], q1, q2);
      out.fill(w[L.b + o + 2], q2, q3); out.fill(w[L.b + o + 3], q3, q3 + plane);
      for (let c = 0; c < C; c++) {
        const A = L.w + (o * C + c) * 9, B = A + C * 9, D = B + C * 9, E = D + C * 9;
        const a0 = w[A], a1 = w[A + 1], a2 = w[A + 2], a3 = w[A + 3], a4 = w[A + 4], a5 = w[A + 5], a6 = w[A + 6], a7 = w[A + 7], a8 = w[A + 8];
        const b0 = w[B], b1 = w[B + 1], b2 = w[B + 2], b3 = w[B + 3], b4 = w[B + 4], b5 = w[B + 5], b6 = w[B + 6], b7 = w[B + 7], b8 = w[B + 8];
        const d0 = w[D], d1 = w[D + 1], d2 = w[D + 2], d3 = w[D + 3], d4 = w[D + 4], d5 = w[D + 5], d6 = w[D + 6], d7 = w[D + 7], d8 = w[D + 8];
        const e0 = w[E], e1 = w[E + 1], e2 = w[E + 2], e3 = w[E + 3], e4 = w[E + 4], e5 = w[E + 5], e6 = w[E + 6], e7 = w[E + 7], e8 = w[E + 8];
        const pc = c * pplane;
        for (let y = 0; y < H; y++) {
          const r0 = pc + y * PW, r1 = r0 + PW, r2 = r1 + PW, row = y * W;
          for (let xx = 0; xx < W; xx++) {
            const v0 = xp[r0 + xx], v1 = xp[r0 + xx + 1], v2 = xp[r0 + xx + 2];
            const v3 = xp[r1 + xx], v4 = xp[r1 + xx + 1], v5 = xp[r1 + xx + 2];
            const v6 = xp[r2 + xx], v7 = xp[r2 + xx + 1], v8 = xp[r2 + xx + 2];
            const i = row + xx;
            out[q0 + i] += a0 * v0 + a1 * v1 + a2 * v2 + a3 * v3 + a4 * v4 + a5 * v5 + a6 * v6 + a7 * v7 + a8 * v8;
            out[q1 + i] += b0 * v0 + b1 * v1 + b2 * v2 + b3 * v3 + b4 * v4 + b5 * v5 + b6 * v6 + b7 * v7 + b8 * v8;
            out[q2 + i] += d0 * v0 + d1 * v1 + d2 * v2 + d3 * v3 + d4 * v4 + d5 * v5 + d6 * v6 + d7 * v7 + d8 * v8;
            out[q3 + i] += e0 * v0 + e1 * v1 + e2 * v2 + e3 * v3 + e4 * v4 + e5 * v5 + e6 * v6 + e7 * v7 + e8 * v8;
          }
        }
      }
    }
    for (; o < O; o++) {                           // 输出通道数不是 4 的倍数时，剩下的逐个算
      const q = o * plane;
      out.fill(w[L.b + o], q, q + plane);
      for (let c = 0; c < C; c++) {
        const A = L.w + (o * C + c) * 9, pc = c * pplane;
        for (let y = 0; y < H; y++) {
          for (let xx = 0; xx < W; xx++) {
            let s = 0;
            for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) s += w[A + ky * 3 + kx] * xp[pc + (y + ky) * PW + xx + kx];
            out[q + y * W + xx] += s;
          }
        }
      }
    }
    for (let i = 0; i < out.length; i++) if (out[i] < 0) out[i] = 0;
    return out;
  }

  function maxpool2(x, C, H, W) {
    const H2 = H >> 1, W2 = W >> 1, out = new Float32Array(C * H2 * W2);
    for (let c = 0; c < C; c++) {
      for (let y = 0; y < H2; y++) {
        for (let xx = 0; xx < W2; xx++) {
          const i = c * H * W + 2 * y * W + 2 * xx;
          out[c * H2 * W2 + y * W2 + xx] = Math.max(x[i], x[i + 1], x[i + W], x[i + W + 1]);
        }
      }
    }
    return out;
  }

  function avgpoolRows(x, C, H, W) {
    const out = new Float32Array(C * W);
    for (let c = 0; c < C; c++) {
      for (let xx = 0; xx < W; xx++) {
        let s = 0;
        for (let y = 0; y < H; y++) s += x[c * H * W + y * W + xx];
        out[c * W + xx] = s / H;
      }
    }
    return out;
  }

  function fc(x, L, relu) {
    const out = new Float32Array(L.out);
    for (let j = 0; j < L.out; j++) {
      let s = weights[L.b + j];
      const row = L.w + j * L.in;
      for (let i = 0; i < L.in; i++) s += weights[row + i] * x[i];
      out[j] = relu && s < 0 ? 0 : s;
    }
    return out;
  }

  function forward(gray) {
    let C = 1, H = meta.height, W = meta.width;
    let x = new Float32Array(H * W);
    for (let i = 0; i < x.length; i++) x[i] = (gray[i] / 255 - meta.mean) / meta.std;
    for (const L of meta.layers) {
      if (L.type === 'conv3x3_relu') { x = conv3x3Relu(x, C, H, W, L); C = L.out; }
      else if (L.type === 'maxpool2') { x = maxpool2(x, C, H, W); H >>= 1; W >>= 1; }
      else if (L.type === 'avgpool_rows') { x = avgpoolRows(x, C, H, W); H = 1; }
      else if (L.type === 'fc_relu') x = fc(x, L, true);
      else if (L.type === 'fc') x = fc(x, L, false);
      else throw new Error('未知层类型 ' + L.type);
    }
    return x; // length × classes 的 logits
  }

  // rgba: Uint8ClampedArray（canvas getImageData 的 data）
  function predict(rgba, w, h) {
    if (!meta) throw new Error('模型还没加载');
    const t0 = performance.now();
    const logits = forward(preprocess(rgba, w, h));
    const n = meta.classes, digits = [];
    let text = '', conf = 1;
    for (let p = 0; p < meta.length; p++) {
      const row = logits.subarray(p * n, p * n + n);
      const mx = Math.max(...row);
      let sum = 0;
      const e = Array.from(row, (v) => { const t = Math.exp(v - mx); sum += t; return t; });
      let best = 0;
      for (let k = 1; k < n; k++) if (e[k] > e[best]) best = k;
      const prob = e[best] / sum;
      text += meta.charset[best];
      conf *= prob;
      digits.push({ char: meta.charset[best], prob });
    }
    return { text, conf, digits, ms: performance.now() - t0 };
  }

  return { load, predict, preprocess, get meta() { return meta; } };
})();

if (typeof module !== 'undefined') module.exports = JppModel;
