// J-PlatPat 验证码助手 —— 内容脚本
//   识别：验证码出现后，用本地模型（model.js，纯 JS）识别并填进输入框；置信度低就换一张。
//   自动提交：只有你亲手操作后弹出的验证码才会自动点 OK（本页的操作，或点开本页的那一下），判错就重新识别再提交，每个对话框最多 3 次。
//   PDF：「PDFダウンロード」出现时自动点；查看器窗口由 page_hook.js 拦下，改成直接下载文件。
//   收集：记录你提交的值和结果（通过 / 输错），交给 background 保存图片并统计模型的实测准确率。
//   可选“自动换图”：对话框已打开时，按随机间隔点“再描画”收集训练图片，到目标数量自动停止。

const CFG = {
  // J-PlatPat 实测结构（2026-10）：Angular Material 对话框，图片为同源 300×60 GIF
  //   OK = button.left，再描画 = #repaint_button，音声再生 = #play_button
  // 这里置为 null 时退回自动定位
  imgSelector: 'mat-dialog-container .image img',
  inputSelector: 'mat-dialog-container input[placeholder="0000"]',
  nonSubmitSelector: '#repaint_button, #play_button',
  // 点“OK”后这么久之内图片被换掉，视为输错
  submitWindowMs: 6000,
  pollMs: 200,
  // 这些按钮不算“提交”（再描画 / 音声再生）
  nonSubmitText: /再描画|再表示|音声|更新|リロード|redraw|refresh|audio|重新|音频/i,
  redrawSelector: '#repaint_button',
  okSelector: 'button.left',
  userActionWindowMs: 15000,  // 对话框出现前这么久之内有你的真实操作，才算“你触发的”，才会自动提交
  autoSubmitDelayMs: 400,
  pdfLinkSelector: 'a#pdfDownloadLink',
  // 点「文献単位PDF」后弹出的确认框：「文献単位PDFの出力には、数分かかる場合があります。」[継続] [キャンセル]
  pdfConfirmSelector: 'pdf-abstract-dialog',
  pdfContinueSelector: '.buttonArea button',   // 第一个按钮是「継続」
  pdfClickWindowMs: 90000,    // 兜底：你最近 90 秒内操作过，出现的「PDFダウンロード」也自动点
  pdfFlowMs: 10 * 60000,      // 由你发起的一次 PDF 下载（含过验证码、服务器生成 PDF 可能要几分钟），10 分钟内都算
  minRedrawGapMs: 1000,   // 自动换图最短间隔，popup 里填更小的值也按这个算
  redrawTimeoutMs: 5000,  // 点“再描画”后这么久新图还没出来，算一次失败（换图本身约 0.8 秒）
  stuckLimit: 3,          // 连续失败几次就判断被限流，自动停止
};

const LOG = (...a) => console.log('[jpp-captcha]', ...a);

const state = {
  enabled: true,
  hash: null,        // 当前显示的验证码图片哈希
  dataUrl: null,
  submitted: null,   // { value, t, hash }
  bound: new WeakSet(),
};

// 自动换图设置（popup 写入 chrome.storage.local）
const auto = { on: false, target: 300, minMs: 1000, maxMs: 1500, done: 0 };
let nextRedrawAt = 0;
let lastClick = null; // { hash, t } 上一次点“再描画”时的图片和时间
let stuck = 0;

// 自动识别设置
const recog = { on: true, minConf: 0.7, redrawLowConf: true, maxRedraw: 3, autoSubmit: true, maxSubmit: 3 };

// PDF 下载设置
//   directPdf：不弹 J-PlatPat 的 PDF 查看器窗口，直接下载文件（拦截由 page_hook.js 在页面环境里完成）
//   autoClick：「PDFダウンロード」按钮出现时自动点（只在你刚亲手操作过之后）
const dl = { directPdf: true, autoClick: true };
const applyDl = () => { document.documentElement.dataset.jppDirectPdf = dl.directPdf ? '1' : '0'; };

chrome.storage.local.get({ enabled: true, auto, recog, dl }, (v) => {
  state.enabled = v.enabled;
  Object.assign(auto, v.auto);
  Object.assign(recog, v.recog);
  Object.assign(dl, v.dl);
  applyDl();
});
applyDl();
chrome.storage.onChanged.addListener((c) => {
  if (c.enabled) state.enabled = c.enabled.newValue;
  if (c.recog) Object.assign(recog, c.recog.newValue);
  if (c.dl) { Object.assign(dl, c.dl.newValue); applyDl(); }
  if (c.auto) {
    if (c.auto.newValue?.on && !c.auto.oldValue?.on) { lastClick = null; stuck = 0; nextRedrawAt = 0; } // 新一轮
    Object.assign(auto, c.auto.newValue);
  }
});

// 停止所有页面的自动换图（由 background 统一改状态，避免各页面互相覆盖计数）
function stopAuto(reason) {
  auto.on = false;
  send({ type: 'stopAuto', reason });
  LOG('自动换图已停止：' + reason);
}

// 多个页面同时换图时，计数由 background 统一发号：领到号才能点“再描画”，领满目标数就全部停止
let ticketPending = false;

async function autoStep(cap) {
  if (!auto.on || !state.hash || ticketPending) return;
  if (cap.input && cap.input.value) return;   // 你在输入时暂停
  if (Date.now() < nextRedrawAt) return;

  // 等上一次换图完成：新图出来了才点下一次；超时没出来算一次失败
  if (lastClick) {
    if (lastClick.hash !== state.hash) {
      stuck = 0;
    } else if (Date.now() - lastClick.t < CFG.redrawTimeoutMs) {
      return; // 新图还在加载
    } else {
      stuck += 1;
      if (stuck >= CFG.stuckLimit) return stopAuto('连续几次换图失败，可能被限流，请稍后再试');
    }
  }

  const btn = cap.container.querySelector(CFG.redrawSelector);
  if (!btn) return;
  ticketPending = true;
  let ticket = null;
  try {
    ticket = await chrome.runtime.sendMessage({ type: 'redrawTicket' });
  } catch (e) {
    LOG('扩展已重新加载，请刷新页面', e);
  } finally {
    ticketPending = false;
  }
  if (!ticket?.ok) return;
  btn.click();
  lastClick = { hash: state.hash, t: Date.now() };
  const min = Math.max(CFG.minRedrawGapMs, auto.minMs);
  const max = Math.max(min, auto.maxMs);
  nextRedrawAt = Date.now() + min + Math.random() * (max - min);
}

function isVisible(el) {
  const r = el.getBoundingClientRect();
  if (r.width === 0 || r.height === 0) return false;
  const s = getComputedStyle(el);
  return s.visibility !== 'hidden' && s.display !== 'none';
}

function looksLikeCaptchaImage(el) {
  if (!isVisible(el)) return false;
  const r = el.getBoundingClientRect();
  const ar = r.width / r.height;
  return r.width >= 100 && ar >= 2.5 && ar <= 8;
}

// 返回 { img, input, container } 或 null
function findCaptcha() {
  if (CFG.imgSelector) {
    const img = document.querySelector(CFG.imgSelector);
    if (!img || !isVisible(img)) return null;
    const input = CFG.inputSelector ? document.querySelector(CFG.inputSelector) : null;
    const container = img.closest('mat-dialog-container') || (input ? commonAncestor(img, input) : img.parentElement);
    return { img, input, container };
  }
  return findCaptchaHeuristic();
}

function findCaptchaHeuristic() {
  const inputs = [...document.querySelectorAll('input')].filter((i) =>
    isVisible(i) &&
    (i.placeholder === '0000' || i.maxLength === 4 ||
     /captcha|画像|認証|auth/i.test(`${i.name} ${i.id} ${i.className} ${i.getAttribute('aria-label') || ''}`)));
  for (const input of inputs) {
    let c = input.parentElement;
    for (let d = 0; c && d < 8; d++, c = c.parentElement) {
      const img = [...c.querySelectorAll('img, canvas')].find(looksLikeCaptchaImage);
      if (img) return { img, input, container: c };
    }
  }
  return null;
}

function commonAncestor(a, b) {
  let c = a.parentElement;
  while (c && !c.contains(b)) c = c.parentElement;
  return c || a.parentElement;
}

// 不重新请求图片（重新请求会让服务器换一张验证码），只从已渲染的元素取像素
function grab(el) {
  try {
    if (el.tagName === 'CANVAS') return el.toDataURL('image/png');
    if (!el.complete || !el.naturalWidth) return null;
    if (el.src.startsWith('data:image/')) return el.src;       // 原样保存，不重新压缩
    const cv = document.createElement('canvas');
    cv.width = el.naturalWidth;
    cv.height = el.naturalHeight;
    cv.getContext('2d').drawImage(el, 0, 0);
    return cv.toDataURL('image/png');
  } catch (e) {
    LOG('取图失败（可能是跨域图片），请导出 DOM 发给开发者', e);
    return null;
  }
}

// ---------------- 自动识别 ----------------

let modelPromise = null;
function ensureModel() {
  if (!modelPromise) {
    modelPromise = JppModel.load(chrome.runtime.getURL('model/')).then((m) => {
      LOG(`模型已加载（${m.trained_at || m.exported_at}，${m.samples || '?'} 张训练）`);
      return m;
    }).catch((e) => { modelPromise = null; throw e; });
  }
  return modelPromise;
}

// 本对话框里模型最近一次的识别结果
const fill = { hash: null, text: null, conf: 0, redraws: 0, busy: false };

// ---------------- 自动提交 ----------------
// 只有“你亲手操作”之后弹出的验证码才会自动点 OK：对话框出现前 15 秒内要有真实的鼠标/键盘操作（isTrusted）。
// 每个对话框最多自动提交 maxSubmit 次；插件自己从不点下载按钮。这样它只是省掉你输入和点 OK，没法被拿去挂脚本批量下载。
//
// 你的操作记两份：
//   lastUserAction   本页面里的
//   sharedUserAction 所有 J-PlatPat 页面共用（chrome.storage.local），只取“打开本页之前”的那一次。
//                    在列表页点开文献（新标签页）、刷新、整页跳转后，新页面自己没有记录，验证码又是一打开就弹，
//                    靠它才能认出是你触发的。本页打开之后别的标签页里的操作不算。
const PAGE_START = performance.timeOrigin;   // 本页面开始加载的时间
let lastUserAction = 0;
let sharedUserAction = 0;
let lastSharedWrite = 0;
for (const t of ['click', 'keydown']) {
  document.addEventListener(t, (e) => {
    if (!e.isTrusted) return;
    const now = Date.now();
    lastUserAction = now;
    if (now - lastSharedWrite >= 1000) {     // 最多每秒写一次
      lastSharedWrite = now;
      try { chrome.storage.local.set({ lastUserAction: now }); } catch { /* 扩展已重新加载 */ }
    }
  }, true);
}
const takeShared = (t) => { if (t && t <= PAGE_START && t > sharedUserAction) sharedUserAction = t; };
chrome.storage.local.get({ lastUserAction: 0 }, (v) => takeShared(v.lastUserAction));
chrome.storage.onChanged.addListener((c) => { if (c.lastUserAction) takeShared(c.lastUserAction.newValue); });

// ms 之内有没有你的操作：'page' 本页里的，'before' 打开本页之前的，null 没有
function userActedWithin(ms) {
  const now = Date.now();
  if (now - lastUserAction < ms) return 'page';
  if (now - sharedUserAction < ms) return 'before';
  return null;
}

const session = { active: false, byUser: false, submits: 0 };  // 当前这个验证码对话框

function startSessionIfNeeded() {
  if (session.active) return;
  session.active = true;
  const by = userActedWithin(CFG.userActionWindowMs);
  session.byUser = !!by;
  session.submits = 0;
  LOG(by === 'page' ? '验证码对话框出现（由你在本页的操作触发，可以自动提交）'
    : by === 'before' ? '验证码对话框出现（由你打开本页的操作触发，可以自动提交）'
    : '验证码对话框出现（15 秒内没有你的操作，只填入不提交）');
}

function endSession() {
  session.active = false;
  session.byUser = false;
  session.submits = 0;
}

function autoSubmit(cap, h, p, info) {
  if (!recog.autoSubmit || auto.on) return false;
  if (!session.byUser) {
    showBadge(cap, `🤖 ${info}，确认无误后点 OK<br><span style="color:#777">15 秒内没检测到你的操作，所以没有自动提交</span>`, '#15803d');
    return true;
  }
  if (session.submits >= recog.maxSubmit) {
    showBadge(cap, `🤖 ${info}，<b>已自动提交 ${session.submits} 次仍未通过，请自己核对后点 OK</b>`, '#b91c1c');
    return true;
  }
  const ok = cap.container.querySelector(CFG.okSelector);
  if (!ok || cap.input?.value !== p.text) return false;
  showBadge(cap, `🤖 ${info}，自动提交（第 ${session.submits + 1} 次）`, '#15803d');
  // 稍等一下再点：让页面处理完输入事件，你也能看到填了什么
  setTimeout(() => {
    if (state.hash !== h || cap.input?.value !== p.text || !document.contains(ok)) return;
    session.submits += 1;
    LOG(`自动提交 ${p.text}（第 ${session.submits} 次）`);
    ok.click();   // 会经过 bind() 里的点击监听 → onSubmit()，照常记录结果
  }, CFG.autoSubmitDelayMs + Math.random() * 300);
  return true;
}

function imageDataOf(el) {
  const w = el.naturalWidth || el.width, h = el.naturalHeight || el.height;
  const cv = document.createElement('canvas');
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext('2d');
  ctx.drawImage(el, 0, 0);
  return ctx.getImageData(0, 0, w, h);
}

// Angular 的输入框要用原生 setter + input 事件，页面才会认到值
function setInputValue(input, v) {
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, v);
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
}

function showBadge(cap, html, color) {
  if (!cap.input) return;
  let b = cap.container.querySelector('#jpp-captcha-badge');
  if (!b) {
    b = document.createElement('div');
    b.id = 'jpp-captcha-badge';
    b.style.cssText = 'font:12px/1.4 system-ui,"Microsoft YaHei",sans-serif;margin:4px 0 0;';
    cap.input.insertAdjacentElement('afterend', b);
  }
  b.style.color = color;
  b.innerHTML = html;
}

// afterWrong：上一张刚被判错，输入框里留的是旧答案，可以直接覆盖
async function recognize(cap, h, afterWrong = false) {
  if (fill.busy) return;
  fill.busy = true;
  try {
    await ensureModel();
    await new Promise((r) => setTimeout(r, 0));          // 先让页面把新图画出来
    if (state.hash !== h) return;                          // 识别前图片又换了
    const p = JppModel.predict(imageDataOf(cap.img).data, cap.img.naturalWidth, cap.img.naturalHeight);
    if (state.hash !== h) return;
    const input = cap.input;
    const prev = fill.text;
    fill.hash = h;
    fill.text = p.text;
    fill.conf = p.conf;
    const info = `识别 <b>${p.text}</b> · 置信度 ${p.conf.toFixed(2)} · ${Math.round(p.ms)} ms`;

    if (p.conf < recog.minConf && recog.redrawLowConf && fill.redraws < recog.maxRedraw) {
      const btn = cap.container.querySelector(CFG.redrawSelector);
      if (btn) {
        fill.redraws += 1;
        showBadge(cap, `🤖 ${info}，把握不大，换一张（第 ${fill.redraws} 次）`, '#b45309');
        LOG('置信度低，换一张', p);
        btn.click();
        return;
      }
    }
    // 你已经自己输入了就不覆盖（除非输入框里是上一次自动填的，或上一张刚被判错）
    if (input && (afterWrong || !input.value || input.value === prev)) {
      setInputValue(input, p.text);
      input.focus();
    }
    send({ type: 'filled' });
    if (p.conf >= recog.minConf) {
      if (!autoSubmit(cap, h, p, info)) showBadge(cap, `🤖 ${info}，确认无误后点 OK`, '#15803d');
    } else {
      showBadge(cap, `🤖 ${info}，<b>把握不大，请核对后点 OK</b>`, '#b91c1c');   // 没把握时不自动提交
    }
    LOG('已填入', p);
  } catch (e) {
    showBadge(cap, '🤖 识别失败：' + e.message, '#b91c1c');
    LOG('识别失败', e);
  } finally {
    fill.busy = false;
  }
}

function hash(s) { // FNV-1a
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return (h >>> 0).toString(16);
}

function send(msg) {
  try { chrome.runtime.sendMessage(msg); } catch (e) { LOG('扩展已重新加载，请刷新页面', e); }
}

function onSubmit(input) {
  const value = (input?.value || '').replace(/\D/g, '');
  if (!state.dataUrl || !value) return;
  state.submitted = { value, t: Date.now(), hash: state.hash };
  // 模型对这张图的识别结果（用来统计实测准确率）
  const model = fill.hash === state.hash ? fill.text : null;
  // 先把图片和输入值交给 background 暂存，防止验证通过后页面跳转导致内容脚本来不及上报
  send({ type: 'pending', hash: state.hash, dataUrl: state.dataUrl, value, model });
  LOG('已提交', value);
}

function bind(cap) {
  if (cap.input && !state.bound.has(cap.input)) {
    state.bound.add(cap.input);
    cap.input.addEventListener('keydown', (e) => { if (e.key === 'Enter') onSubmit(cap.input); }, true);
  }
  if (cap.container && !state.bound.has(cap.container)) {
    state.bound.add(cap.container);
    cap.container.addEventListener('click', (e) => {
      const btn = e.target.closest('button, [role=button], input[type=submit], input[type=button], a');
      if (!btn || !cap.container.contains(btn)) return;
      if (CFG.nonSubmitSelector && btn.matches(CFG.nonSubmitSelector)) return;
      const label = `${btn.textContent || ''} ${btn.value || ''} ${btn.getAttribute('aria-label') || ''}`;
      if (CFG.nonSubmitText.test(label)) return;
      onSubmit(cap.input || cap.container.querySelector('input'));
    }, true);
  }
}

function recentSubmit() {
  const s = state.submitted;
  return s && s.hash === state.hash && Date.now() - s.t < CFG.submitWindowMs ? s : null;
}

// ---------------- PDF 下载 ----------------

// page_hook.js 拦下查看器窗口后，把 PDF 地址发过来，交给 background 用 chrome.downloads 下载
window.addEventListener('message', (e) => {
  if (e.source !== window || e.origin !== location.origin) return;
  const m = e.data;
  if (m?.source !== 'jpp-helper' || m.type !== 'downloadPdf' || !Array.isArray(m.items)) return;
  const items = m.items.filter((x) => typeof x.url === 'string' && new URL(x.url).origin === location.origin);
  if (!items.length) return;
  LOG(`直接下载 PDF ${items.length} 个`, items.map((x) => x.url));
  send({ type: 'downloadPdf', items });
});

// 一次“由你发起的 PDF 下载”开始的时间：你点「文献単位PDF」后确认框被继续（自动或手动）时记下
let pdfFlowStarted = 0;
document.addEventListener('click', (e) => {
  if (e.isTrusted && e.target.closest?.(CFG.pdfConfirmSelector)) pdfFlowStarted = Date.now();   // 你自己点了「継続」
}, true);

// 确认框「文献単位PDFの出力には…」出现时自动点「継続」。只在你刚亲手点了「文献単位PDF」时才点；每个确认框只处理一次
function autoContinuePdf() {
  if (!dl.autoClick) return;
  const dlg = document.querySelector(CFG.pdfConfirmSelector);
  if (!dlg || dlg.dataset.jppClicked) return;
  dlg.dataset.jppClicked = '1';
  if (Date.now() - lastUserAction > CFG.userActionWindowMs) {
    LOG('PDF 确认框出现了，但你最近没有操作，不自动继续');
    return;
  }
  const btn = dlg.querySelector(CFG.pdfContinueSelector);
  if (!btn) return;
  pdfFlowStarted = Date.now();
  LOG('自动点「継続」');
  setTimeout(() => btn.click(), 150);   // 等对话框动画结束
}

// 「PDFダウンロード」出现时自动点。只在这次下载是你发起的时候才点；每个按钮只点一次
function autoClickPdfLink() {
  if (!dl.autoClick) return;
  const link = document.querySelector(CFG.pdfLinkSelector);
  if (!link || link.dataset.jppClicked) return;
  link.dataset.jppClicked = '1';
  const byUser = Date.now() - pdfFlowStarted < CFG.pdfFlowMs || Date.now() - lastUserAction < CFG.pdfClickWindowMs;
  if (!byUser) {
    LOG('「PDFダウンロード」出现了，但不是你刚发起的下载，不自动点');
    return;
  }
  LOG('自动点「PDFダウンロード」');
  link.click();
}

function tick() {
  autoContinuePdf();
  autoClickPdfLink();
  // 注意：state.enabled 只控制“保存图片”（在 background 里判断），识别和统计一直工作
  const cap = findCaptcha();

  if (!cap) {
    // 实测：点 OK 后对话框还在，图片元素会先消失约 0.8 秒再换成新图。
    // 所以只要对话框还在，就当作“换图中”，不能判定为通过。
    // 只认验证码对话框（有 0000 输入框的那个）；「継続」确认框等其它弹窗也是 mat-dialog-container，不能算
    if (document.querySelector(CFG.inputSelector || 'mat-dialog-container')) return;
    fill.redraws = 0;
    endSession();
    if (auto.on && state.hash) LOG('本页验证码对话框已关闭，本页暂停自动换图（其他页面不受影响）');
    if (state.hash) {
      // 验证码区域消失：提交过 → 通过；没提交过（直接关掉）→ 未标注
      const s = recentSubmit();
      if (s) send({ type: 'resolve', hash: state.hash, result: 'ok' });
      else send({ type: 'save', hash: state.hash, dataUrl: state.dataUrl, kind: 'unlabeled' });
      state.hash = state.dataUrl = state.submitted = null;
    }
    return;
  }

  bind(cap);
  const url = grab(cap.img);
  if (!url) return;
  const h = hash(url);
  if (h === state.hash) { autoStep(cap); return; }

  let afterWrong = false;
  if (state.hash) {
    // 图片换了：提交过 → 输错；没提交 → 点了再描画
    const s = recentSubmit();
    afterWrong = !!s;
    if (s) send({ type: 'resolve', hash: state.hash, result: 'wrong' });
    else send({ type: 'save', hash: state.hash, dataUrl: state.dataUrl, kind: 'unlabeled' });
  }
  state.hash = h;
  state.dataUrl = url;
  state.submitted = null;
  LOG('发现验证码', h);
  startSessionIfNeeded();
  // 自动换图收集时不识别（省 CPU，也避免和换图互相干扰）
  if (recog.on && !auto.on && cap.img.tagName === 'IMG') recognize(cap, h, afterWrong);
}

setInterval(tick, CFG.pollMs);

// popup 的“导出验证码区域 HTML”
chrome.runtime.onMessage.addListener((msg, _sender, reply) => {
  if (msg.type !== 'dumpDom') return;
  if (window !== window.top && !document.querySelector('input, img, canvas')) return;
  const cap = findCaptcha();
  let root;
  if (cap) {
    // 往上多取几层，尽量包含整个对话框
    root = cap.container;
    for (let i = 0; i < 3 && root.parentElement && root.parentElement !== document.body; i++) root = root.parentElement;
  } else {
    // 自动定位失败：导出对话框层，没有就导出整个 body（去掉脚本和样式）
    root = document.querySelector('.cdk-overlay-container, [role=dialog], .modal') || document.body;
  }
  const clone = root.cloneNode(true);
  clone.querySelectorAll('script, style, svg, noscript').forEach((n) => n.remove());
  const html = clone.outerHTML.replace(/src="data:image\/[^"]{200,}"/g, 'src="data:image/...(省略)"');
  send({ type: 'saveDom', html, url: location.href, found: !!cap });
  reply({ found: !!cap });
});
