// J-PlatPat 验证码助手 —— 运行在页面自己的 JS 环境里（manifest 里 world: "MAIN"，document_start）
//
// J-PlatPat 点「PDFダウンロード」时的流程（2026-10 实测）：
//   1. 把要显示的 PDF 列表写进 localStorage.p0205Param
//   2. window.open('/p0205', …) 弹出一个查看器小窗口，查看器再从 localStorage 读列表
// 这里把第 2 步拦下来：不弹窗口，直接把 p0205Param 里的 PDF 地址交给插件下载。
// 是否拦截由 content.js 通过 <html data-jpp-direct-pdf="1"> 告诉我们（页面环境里读不到 chrome.storage）。

(() => {
  const origOpen = window.open;

  function pdfList() {
    let p;
    try { p = JSON.parse(localStorage.getItem('p0205Param') || 'null'); } catch { return []; }
    if (!p) return [];
    const list = [];
    for (const d of p.optPdfMainArray?.optPmDivideArray || []) {
      if (d.optPmDivideUrl) list.push({ url: d.optPmDivideUrl, seq: d.optPmDivideSeq, doc: p.optPdfMainArray.optPdfMainNo, main: true });
    }
    for (const sub of p.optPdfSubArray || []) {
      for (const d of sub.optPsDivideArray || []) {
        if (d.optPsDivideUrl) list.push({ url: d.optPsDivideUrl, seq: d.optPsDivideSeq, doc: sub.optPdfPsDocKey, main: false });
      }
    }
    return list.map((x) => ({ ...x, url: new URL(x.url, location.origin).href }));
  }

  window.open = function (url, ...rest) {
    try {
      const path = new URL(String(url), location.origin).pathname;
      if (path === '/p0205' && document.documentElement.dataset.jppDirectPdf === '1') {
        const list = pdfList();
        if (list.length) {
          localStorage.removeItem('p0205Param');   // 查看器读完也会删掉，这里保持一致
          window.postMessage({ source: 'jpp-helper', type: 'downloadPdf', items: list }, location.origin);
          return null;                             // 实测页面不使用返回值
        }
      }
    } catch (e) {
      console.warn('[jpp-captcha] 拦截 PDF 窗口失败，按原样打开', e);
    }
    return origOpen.call(this, url, ...rest);
  };
})();
