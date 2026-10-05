// J-PlatPat 验证码助手 —— 后台：暂存待判定的样本，保存到 下载目录/jpp_captcha/，统计模型实测准确率
//   ok/        验证通过，文件名开头就是正确答案       例：8621_20261005-183012.png
//   wrong/     提交后被刷新（输错），文件名开头是输入值，不是答案
//   unlabeled/ 点了再描画或直接关闭，没有答案
//   debug/     导出的 DOM

const ROOT = 'jpp_captcha';
const PENDING_TIMEOUT_MS = 10000; // 提交后页面直接跳走、没等到结果时，按“通过”处理

const pending = new Map(); // hash -> { dataUrl, value, timer }
const savedHashes = new Set();

function stamp() {
  const d = new Date();
  const p = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}-${p(d.getMilliseconds(), 3)}`;
}

function extOf(dataUrl) {
  const m = /^data:image\/(png|jpeg|jpg|gif|webp|bmp)/.exec(dataUrl);
  return m ? (m[1] === 'jpeg' ? 'jpg' : m[1]) : 'png';
}

// 多个页面同时发消息时，storage 的“读 → 改 → 写”必须排队，否则计数会互相覆盖
let queue = Promise.resolve();
function serial(fn) {
  const p = queue.then(fn);
  queue = p.catch(() => {});
  return p;
}

function bump(kind) {
  return serial(async () => {
    const { counts = { ok: 0, wrong: 0, unlabeled: 0 } } = await chrome.storage.local.get('counts');
    counts[kind] = (counts[kind] || 0) + 1;
    await chrome.storage.local.set({ counts });
  });
}

// 自动换图发号：所有页面共用一个进度，领满目标数就全部停止
function redrawTicket() {
  return serial(async () => {
    const { auto } = await chrome.storage.local.get('auto');
    if (!auto?.on) return { ok: false };
    if (auto.done >= auto.target) {
      await chrome.storage.local.set({ auto: { ...auto, on: false }, autoStopReason: `已达到目标 ${auto.target} 张` });
      return { ok: false };
    }
    await chrome.storage.local.set({ auto: { ...auto, done: auto.done + 1 } });
    return { ok: true };
  });
}

function stopAuto(reason) {
  return serial(async () => {
    const { auto } = await chrome.storage.local.get('auto');
    if (auto?.on) await chrome.storage.local.set({ auto: { ...auto, on: false }, autoStopReason: reason });
  });
}

// “保存图片”开关（popup 的“启用收集”）。关掉后仍然识别、仍然统计准确率，只是不存图
let collect = true;
chrome.storage.local.get({ enabled: true }, (v) => { collect = v.enabled; });
chrome.storage.onChanged.addListener((c) => { if (c.enabled) collect = c.enabled.newValue; });

function save(hash, dataUrl, kind, value) {
  if (!collect || !dataUrl || savedHashes.has(hash)) return;
  savedHashes.add(hash);
  const name = value ? `${value}_${stamp()}` : stamp();
  chrome.downloads.download({
    url: dataUrl,
    filename: `${ROOT}/${kind}/${name}.${extOf(dataUrl)}`,
    conflictAction: 'uniquify',
    saveAs: false,
  });
  bump(kind);
}

// ---------------- PDF 直接下载 ----------------
const PDF_DIR = 'J-PlatPat';

// "JPA 2001330771-000000.pdf" → "JP2001-330771A.pdf"；认不出的格式就保留原文件名（空格换成下划线）
function pdfName(url, multi, seq) {
  const base = decodeURIComponent(new URL(url).pathname.split('/').pop() || 'document.pdf');
  const m = /^JP([A-Z])\s*(\d{4})(\d{6})-\d+\.pdf$/i.exec(base);
  let name = m ? `JP${m[2]}-${m[3]}${m[1].toUpperCase()}` : base.replace(/\.pdf$/i, '').replace(/[\s\\/:*?"<>|]+/g, '_');
  if (multi && seq) name += `_${seq}`;
  return `${name}.pdf`;
}

function downloadPdfs(items) {
  const mainParts = items.filter((x) => x.main).length;
  for (const it of items) {
    chrome.downloads.download({
      url: it.url,
      filename: `${PDF_DIR}/${pdfName(it.url, it.main ? mainParts > 1 : true, it.seq)}`,
      conflictAction: 'uniquify',
      saveAs: false,
    });
    bump('pdf');
  }
}

function resolve(hash, result) {
  const p = pending.get(hash);
  if (!p) return;
  clearTimeout(p.timer);
  pending.delete(hash);
  save(hash, p.dataUrl, result, p.value);
  // 模型的实测准确率：以 J-PlatPat 的判定为准
  if (p.model) {
    if (result === 'ok') bump(p.value === p.model ? 'model_ok' : 'model_wrong');   // 通过了但值不是模型给的 → 模型错了、你改对了
    else if (p.value === p.model) bump('model_wrong');                               // 用模型的答案被判错
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  switch (msg.type) {
    case 'redrawTicket':
      redrawTicket().then(reply, () => reply({ ok: false }));
      return true; // 异步回复
    case 'stopAuto':
      stopAuto(msg.reason);
      break;
    case 'pending': {
      const old = pending.get(msg.hash);
      if (old) clearTimeout(old.timer); // 同一张图重复提交，以最后一次为准
      const timer = setTimeout(() => resolve(msg.hash, 'ok'), PENDING_TIMEOUT_MS);
      pending.set(msg.hash, { dataUrl: msg.dataUrl, value: msg.value, model: msg.model, timer });
      break;
    }
    case 'filled':
      bump('model_filled');
      break;
    case 'downloadPdf':
      if (Array.isArray(msg.items)) {
        downloadPdfs(msg.items.filter((x) => /^https:\/\/[^/]*j-platpat\.inpit\.go\.jp\//.test(x.url)));
      }
      break;
    case 'resolve':
      resolve(msg.hash, msg.result);
      break;
    case 'save':
      if (!pending.has(msg.hash)) save(msg.hash, msg.dataUrl, msg.kind);
      break;
    case 'saveDom': {
      const html = `<!-- ${msg.url}\n     自动定位: ${msg.found ? '成功' : '失败'} -->\n${msg.html}`;
      chrome.downloads.download({
        url: 'data:text/html;charset=utf-8,' + encodeURIComponent(html),
        filename: `${ROOT}/debug/dom_${stamp()}.html`,
        saveAs: false,
      });
      break;
    }
  }
});
