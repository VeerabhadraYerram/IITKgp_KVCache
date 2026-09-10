// app.js — Chapter 1: "What does an AI mean by 'remembering'?"
// Implements typewriter animation on hero and dynamic interactive choice dilemma.

// ─── Typewriter Animation ──────────────────────────────────────
function initTypewriter() {
  const titleEl = document.getElementById('typewriter-title');
  const subEl = document.getElementById('typewriter-sub');
  const promptEl = document.getElementById('scroll-prompt');

  if (!titleEl || !subEl) return;

  const titleText = `What does an AI actually mean by "remembering"?`;
  const subText = `Does it keep a perfect recording of everything you said — or let the past quietly reshape it?`;

  let titleIdx = 0;
  let subIdx = 0;

  // Render cursor helper
  const renderCaret = () => '<span class="caret"></span>';

  function typeTitle() {
    if (titleIdx < titleText.length) {
      titleEl.innerHTML = titleText.slice(0, titleIdx + 1) + renderCaret();
      titleIdx++;
      setTimeout(typeTitle, 40);
    } else {
      titleEl.innerHTML = titleText; // Remove caret from title
      subEl.innerHTML = renderCaret();
      setTimeout(typeSub, 300);
    }
  }

  function typeSub() {
    if (subIdx < subText.length) {
      subEl.innerHTML = subText.slice(0, subIdx + 1) + renderCaret();
      subIdx++;
      setTimeout(typeSub, 30);
    } else {
      subEl.innerHTML = subText; // Keep clean without caret or with static punctuation
      if (promptEl) {
        promptEl.style.opacity = '1';
      }
    }
  }

  // Start typewriter after a short initial pause
  setTimeout(typeTitle, 400);
}

// ─── Chapter 1: Dilemma Widget State Handler ───────────────────
function initDilemmaWidget() {
  const choices = {
    legal: null,
    chat: null
  };

  const buttons = document.querySelectorAll('.choice-btn');
  const summaryBox = document.getElementById('summary-box');
  const summaryLine = document.getElementById('summary-line-text');
  const summaryExplanation = document.getElementById('summary-explanation-text');
  const cardVerbatim = document.getElementById('card-verbatim');
  const cardCompressed = document.getElementById('card-compressed');

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const scenario = btn.dataset.scenario;
      const choice = btn.dataset.choice;

      // Update active state in group
      const siblingBtns = document.querySelectorAll(`.choice-btn[data-scenario="${scenario}"]`);
      siblingBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      choices[scenario] = choice;

      updateSummary();
    });
  });

  function updateSummary() {
    // Show summary once at least one choice is made
    if (!choices.legal && !choices.chat) return;

    summaryBox.style.display = 'block';

    const legalLabel = choices.legal === 'verbatim' ? 'Verbatim Store' : choices.legal === 'compressed' ? 'Compressed State' : 'Not selected';
    const chatLabel = choices.chat === 'verbatim' ? 'Verbatim Store' : choices.chat === 'compressed' ? 'Compressed State' : 'Not selected';

    summaryLine.textContent = `Your Deployment Choices: Legal Audit → ${legalLabel} | Real-Time Chat → ${chatLabel}`;

    let explanation = '';
    if (choices.legal === 'verbatim' && choices.chat === 'compressed') {
      explanation = `You chose exact literal recall for high-stakes legal compliance (where a lost number is a critical failure), but opted for semantic compression in real-time chat (where general gist is enough and memory overhead must stay minimal). You intuitively navigated the exact trade-off at the heart of AI memory.`;
    } else if (choices.legal === 'verbatim' && choices.chat === 'verbatim') {
      explanation = `You prioritized 100% literal accuracy across all workloads, accepting that memory consumption will grow continuously over long conversations.`;
    } else if (choices.legal === 'compressed' && choices.chat === 'compressed') {
      explanation = `You prioritized constant, bounded memory across all workloads, accepting that exact numbers like "$1,450,000" might be recalled only approximately as "around $1.45M".`;
    } else if (choices.legal === 'compressed' && choices.chat === 'verbatim') {
      explanation = `You accepted approximate recall in legal auditing and demanded exact retention in casual chat. This shows how differing task demands force different compromises.`;
    } else {
      explanation = `Make a selection for both scenarios above to see how task requirements dictate memory trade-offs.`;
    }

    summaryExplanation.textContent = explanation;

    // Highlight corresponding response cards subtly
    if (choices.legal === 'verbatim' || choices.chat === 'verbatim') {
      cardVerbatim.style.borderColor = 'rgba(0, 0, 0, 0.6)';
    } else {
      cardVerbatim.style.borderColor = 'var(--border-glass)';
    }

    if (choices.legal === 'compressed' || choices.chat === 'compressed') {
      cardCompressed.style.borderColor = 'rgba(0, 0, 0, 0.6)';
    } else {
      cardCompressed.style.borderColor = 'var(--border-glass)';
    }
  }
}

// ─── Main Initialization ───────────────────────────────────────
function init() {
  initTypewriter();
  initDilemmaWidget();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
