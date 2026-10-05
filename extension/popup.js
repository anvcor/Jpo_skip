const $ = (id) => document.getElementById(id);
const DEFAULT_AUTO = { on: false, target: 300, minMs: 1000, maxMs: 1500, done: 0 };
const DEFAULT_RECOG = { on: true, minConf: 0.7, redrawLowConf: true, maxRedraw: 3, autoSubmit: true, maxSubmit: 3 };

const DEFAULT_DL = { directPdf: true, autoClick: true };

async function refresh() {
  const { enabled = true, counts = {}, auto = DEFAULT_AUTO, autoStopReason = '', recog = DEFAULT_RECOG, dl = DEFAULT_DL } =
    await chrome.storage.local.get(['enabled', 'counts', 'auto', 'autoStopReason', 'recog', 'dl']);
  $('enabled').checked = enabled;
  for (const k of ['ok', 'wrong', 'unlabeled', 'model_filled', 'model_ok', 'model_wrong', 'pdf']) $(k).textContent = counts[k] || 0;
  const d = { ...DEFAULT_DL, ...dl };
  $('directPdf').checked = d.directPdf;
  $('autoClick').checked = d.autoClick;
  const judged = (counts.model_ok || 0) + (counts.model_wrong || 0);
  $('modelAcc').textContent = judged ? `${(100 * (counts.model_ok || 0) / judged).toFixed(1)}%（${judged} 次）` : '–';

  const r = { ...DEFAULT_RECOG, ...recog };
  $('recogOn').checked = r.on;
  $('autoSubmit').checked = r.autoSubmit;
  $('redrawLow').checked = r.redrawLowConf;
  if (document.activeElement?.type !== 'number') {
    $('minConf').value = r.minConf;
    $('maxRedraw').value = r.maxRedraw;
    $('maxSubmit').value = r.maxSubmit;
  }

  $('progress').textContent = `${auto.done} / ${auto.target}`;
  $('autoBtn').textContent = auto.on ? '停止自动换图' : '开始自动换图';
  $('stopReason').textContent = auto.on ? '运行中…在输入框里打字会暂停' : autoStopReason;
  if (!auto.on && document.activeElement?.type !== 'number') {
    $('target').value = auto.target;
    $('minS').value = auto.minMs / 1000;
    $('maxS').value = auto.maxMs / 1000;
  }
}

$('enabled').addEventListener('change', (e) => chrome.storage.local.set({ enabled: e.target.checked }));

async function saveRecog() {
  await chrome.storage.local.set({
    recog: {
      on: $('recogOn').checked,
      minConf: Math.min(1, Math.max(0, Number($('minConf').value) || 0)),
      redrawLowConf: $('redrawLow').checked,
      maxRedraw: Math.min(10, Math.max(0, Math.round(Number($('maxRedraw').value) || 0))),
      autoSubmit: $('autoSubmit').checked,
      maxSubmit: Math.min(10, Math.max(1, Math.round(Number($('maxSubmit').value) || 3))),
    },
  });
}
for (const id of ['recogOn', 'autoSubmit', 'maxSubmit', 'minConf', 'redrawLow', 'maxRedraw']) $(id).addEventListener('change', saveRecog);

for (const id of ['directPdf', 'autoClick']) {
  $(id).addEventListener('change', () => chrome.storage.local.set({ dl: { directPdf: $('directPdf').checked, autoClick: $('autoClick').checked } }));
}

// 显示模型版本
fetch(chrome.runtime.getURL('model/jpp_cnn.json')).then((r) => r.json()).then((m) => {
  $('modelInfo').textContent = `本地模型 · ${m.trained_at || m.exported_at} 训练 · ${m.samples || '?'} 张样本 · 不联网`;
}).catch(() => { $('modelInfo').textContent = '⚠ 找不到模型文件 model/jpp_cnn.bin'; });

$('autoBtn').addEventListener('click', async () => {
  const { auto = DEFAULT_AUTO } = await chrome.storage.local.get('auto');
  if (auto.on) {
    await chrome.storage.local.set({ auto: { ...auto, on: false }, autoStopReason: '已手动停止' });
    return;
  }
  const minS = Math.max(1, Number($('minS').value) || 1);
  const maxS = Math.max(minS, Number($('maxS').value) || 1.5);
  const target = Math.min(2000, Math.max(10, Number($('target').value) || 300));
  await chrome.storage.local.set({
    auto: { on: true, target, minMs: minS * 1000, maxMs: maxS * 1000, done: 0 },
    autoStopReason: '',
  });
});

$('reset').addEventListener('click', async () => {
  await chrome.storage.local.set({ counts: { ok: 0, wrong: 0, unlabeled: 0, model_filled: 0, model_ok: 0, model_wrong: 0, pdf: 0 } });
  refresh();
});

$('dump').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  try {
    const r = await chrome.tabs.sendMessage(tab.id, { type: 'dumpDom' });
    $('msg').textContent = r?.found
      ? '已导出（定位成功）→ jpp_captcha\\debug\\'
      : '已导出（没定位到验证码，导出了整个对话框/页面）→ jpp_captcha\\debug\\';
  } catch {
    $('msg').textContent = '当前页面不是 J-PlatPat，或需要刷新页面后再试';
  }
});

chrome.storage.onChanged.addListener(refresh);
refresh();
