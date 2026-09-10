import fs from 'fs';

async function verifyAndCapture() {
  const res = await fetch('http://127.0.0.1:9222/json/list');
  const tabs = await res.json();
  const page = tabs.find(t => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let id = 1;

  function send(method, params = {}) {
    return new Promise((resolve) => {
      const msgId = id++;
      const handler = (e) => {
        const data = JSON.parse(e.data);
        if (data.id === msgId) {
          ws.removeEventListener('message', handler);
          resolve(data.result);
        }
      };
      ws.addEventListener('message', handler);
      ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  }

  ws.onopen = async () => {
    await send('Page.enable');
    await send('DOM.enable');
    await send('CSS.enable');

    // 1. Desktop Dark Mode Verification
    await send('Emulation.setDeviceMetricsOverride', {
      width: 1280,
      height: 900,
      deviceScaleFactor: 1,
      mobile: false
    });
    await send('Page.navigate', { url: 'http://localhost:8765/' });
    await new Promise(r => setTimeout(r, 2000));

    // Ensure dark mode is active
    await send('Runtime.evaluate', {
      expression: `
        if (!document.body.classList.contains('dark-mode')) {
          document.getElementById('theme-toggle').click();
        }
      `
    });
    await new Promise(r => setTimeout(r, 500));

    // Evaluate contrast check
    const contrastCheck = await send('Runtime.evaluate', {
      expression: `(() => {
        const getStyles = (sel) => {
          const el = document.querySelector(sel);
          if (!el) return null;
          const comp = window.getComputedStyle(el);
          return { color: comp.color, bg: comp.backgroundColor };
        };
        return {
          bodyBg: window.getComputedStyle(document.body).backgroundColor,
          decisionHeading: getStyles('.decision-heading'),
          scenarioPrompt: getStyles('.scenario-prompt'),
          choiceBtn: getStyles('.choice-btn'),
          specTitle: getStyles('.spec-title'),
          specDesc: getStyles('.spec-desc'),
          toyModelTitle: getStyles('.toy-model-title'),
          toyModelBody: getStyles('.toy-model-body'),
          quizQText: getStyles('.quiz-q-text'),
          quizOptionBtn: getStyles('.quiz-option-btn')
        };
      })()`,
      returnByValue: true
    });
    console.log('Contrast Check Computed Styles:', JSON.stringify(contrastCheck.result.value, null, 2));

    // Capture Desktop Chapter 1 (Decision & Playground)
    await send('Runtime.evaluate', {
      expression: `document.getElementById('chapter-1').scrollIntoView({ block: 'start' });`
    });
    await new Promise(r => setTimeout(r, 800));
    const shotCh1 = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('C:/Users/Abhilash/.gemini/antigravity-ide/brain/052e5ba6-911e-4749-8a6f-e1989eb38ae5/shot_fixed_dark_ch1.png', Buffer.from(shotCh1.data, 'base64'));
    console.log('Saved shot_fixed_dark_ch1.png');

    // Capture Desktop Chapter 6 (Spec boxes & Toy model)
    await send('Runtime.evaluate', {
      expression: `document.querySelector('.dual-spec-card').scrollIntoView({ block: 'center' });`
    });
    await new Promise(r => setTimeout(r, 800));
    const shotCh6 = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('C:/Users/Abhilash/.gemini/antigravity-ide/brain/052e5ba6-911e-4749-8a6f-e1989eb38ae5/shot_fixed_dark_ch6.png', Buffer.from(shotCh6.data, 'base64'));
    console.log('Saved shot_fixed_dark_ch6.png');

    // Capture Sixty-Second Test
    await send('Runtime.evaluate', {
      expression: `document.getElementById('quiz-section').scrollIntoView({ block: 'center' });`
    });
    await new Promise(r => setTimeout(r, 800));
    const shotQuiz = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('C:/Users/Abhilash/.gemini/antigravity-ide/brain/052e5ba6-911e-4749-8a6f-e1989eb38ae5/shot_fixed_dark_quiz.png', Buffer.from(shotQuiz.data, 'base64'));
    console.log('Saved shot_fixed_dark_quiz.png');

    // 2. Mobile Phone (iPhone 14 / Pixel: 390 x 844)
    await send('Emulation.setDeviceMetricsOverride', {
      width: 390,
      height: 844,
      deviceScaleFactor: 2,
      mobile: true
    });
    await new Promise(r => setTimeout(r, 1000));

    // Check mobile overflow and metrics
    const mobileCheck = await send('Runtime.evaluate', {
      expression: `(() => {
        const topbar = document.getElementById('mobile-topbar');
        const docWidth = document.documentElement.offsetWidth;
        const scrollWidth = document.documentElement.scrollWidth;
        const navRail = document.getElementById('chapter-nav');
        return {
          mobileTopbarVisible: window.getComputedStyle(topbar).display !== 'none',
          navRailHidden: window.getComputedStyle(navRail).display === 'none',
          hasHorizontalOverflow: scrollWidth > docWidth,
          docWidth,
          scrollWidth
        };
      })()`,
      returnByValue: true
    });
    console.log('Mobile Layout Check:', JSON.stringify(mobileCheck.result.value, null, 2));

    // Capture Mobile Hero
    await send('Runtime.evaluate', {
      expression: `window.scrollTo(0, 0);`
    });
    await new Promise(r => setTimeout(r, 800));
    const shotMobHero = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('C:/Users/Abhilash/.gemini/antigravity-ide/brain/052e5ba6-911e-4749-8a6f-e1989eb38ae5/shot_mobile_phone_hero.png', Buffer.from(shotMobHero.data, 'base64'));
    console.log('Saved shot_mobile_phone_hero.png');

    // Capture Mobile Chapter 1 (Decision & Prompt playground)
    await send('Runtime.evaluate', {
      expression: `document.querySelector('.decision-section').scrollIntoView({ block: 'start' });`
    });
    await new Promise(r => setTimeout(r, 800));
    const shotMobCh1 = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync('C:/Users/Abhilash/.gemini/antigravity-ide/brain/052e5ba6-911e-4749-8a6f-e1989eb38ae5/shot_mobile_phone_ch1.png', Buffer.from(shotMobCh1.data, 'base64'));
    console.log('Saved shot_mobile_phone_ch1.png');

    ws.close();
    process.exit(0);
  };
}

verifyAndCapture().catch(err => {
  console.error(err);
  process.exit(1);
});
