/* ==========================================================================
   ShopBot widget — a large, ChatGPT-style assistant you can drop on any site
   --------------------------------------------------------------------------
   Vanilla JS. No framework, no build step, no third-party runtime deps.

   The whole UI lives in a Shadow DOM, so the host page cannot style it and it
   cannot leak styles back. It is host-agnostic:

     ShopBot.init({
       apiBase: 'http://localhost:5000',   // optional; defaults to the origin
       host: {
         navigate: function (url) { ... },  // called when a reply points somewhere
         getContext: function () { ... }    // {page, asin, query, filters, cart_count, order_id}
       }
     })

   With no host callbacks it still works alone: it navigates by setting
   `location.hash` from the site's page templates (GET /api/site) and builds a
   page context from the current hash.

   Endpoints:  GET /api/site · GET /api/health · POST /api/chat ·
               POST /api/transcribe
   ========================================================================== */

(function () {
  'use strict';

  /* ------------------------------------------------------------- constants */

  var SCRIPT_URL = (document.currentScript && document.currentScript.src) || '';
  var CSS_URL = SCRIPT_URL
    ? SCRIPT_URL.replace(/shopbot\.js(\?.*)?$/i, 'shopbot.css')
    : 'shopbot/shopbot.css';

  var DEFAULT_API_BASE =
    (location.protocol === 'http:' || location.protocol === 'https:')
      ? location.origin
      : 'http://localhost:5000';

  var STORAGE_LANG = 'shopbot_lang';
  var STORAGE_MODE = 'shopbot_mode';
  var STORAGE_CONVS = 'shopbot_conversations';
  var STORAGE_ACTIVE = 'shopbot_active';
  var SESSION_ID = 'shopbot_session_id';

  var MAX_CONVERSATIONS = 30;
  var MAX_MESSAGES = 60;
  var HEALTH_INTERVAL = 45000;

  // Words that only phrase a request, not describe a product. The storefront
  // search needs every word to appear in a title, so "show me brush" would
  // otherwise match nothing.
  var SEARCH_STOPWORDS = {
    i: 1, a: 1, an: 1, the: 1, want: 1, need: 1, looking: 1, for: 1, find: 1,
    show: 1, me: 1, can: 1, could: 1, you: 1, please: 1, some: 1, something: 1,
    give: 1, get: 1, buy: 1, do: 1, have: 1, has: 1, is: 1, are: 1, of: 1,
    to: 1, in: 1, on: 1, with: 1, and: 1, or: 1, my: 1, any: 1,
    // Arabic request words
    '\u0623\u0631\u064a\u062f': 1, '\u0627\u0631\u064a\u062f': 1, '\u0623\u0628\u062d\u062b': 1,
    '\u0627\u0628\u062d\u062b': 1, '\u0623\u0631\u0646\u064a': 1, '\u0627\u0631\u0646\u064a': 1,
    '\u0647\u0644': 1, '\u0645\u0646': 1, '\u0641\u064a': 1, '\u0639\u0644\u0649': 1,
    '\u0639\u0646': 1, '\u0645\u0639': 1, '\u0644\u064a': 1
  };

  function cleanSearchQuery(text) {
    var tokens = String(text || '').toLowerCase().split(/\s+/);
    var kept = [];
    tokens.forEach(function (token) {
      var word = token.replace(/[^\u00c0-\u024f\u0600-\u06ffa-z0-9]/g, '');
      if (word && !SEARCH_STOPWORDS[word]) { kept.push(word); }
    });
    return kept.length ? kept.join(' ') : (text || '').trim();
  }

  var FALLBACK_PAGES = {
    home: '#/',
    search: '#/search?q={q}&asins={asins}&min_price={min_price}&max_price={max_price}&category={category}&sort={sort}',
    category: '#/category/{category}',
    product: '#/product/{asin}',
    compare: '#/compare?asins={asins}',
    cart: '#/cart',
    orders: '#/orders?order_id={order_id}',
    help: '#/help/{topic}'
  };

  /* ------------------------------------------------------------------ i18n */

  var I18N = {
    en: {
      launcher: 'Ask {assistant}',
      greeting_title: "Hi, I'm {assistant}",
      greeting_text: 'Ask me to find products, compare two or three, or ask about warranty, returns, delivery, cash on delivery and orders. You can also speak your question.',
      chip_hair_dryer: 'Find a hair dryer under $30',
      chip_hair_brush: 'Show me hair brushes',
      chip_cod: 'Is there cash on delivery?',
      chip_order: 'Where is my order ORD-10025?',
      placeholder: 'Ask about products, comparisons, policies or orders…',
      send: 'Send',
      mic: 'Voice input',
      new_chat: 'New chat',
      hide_conversations: 'Hide conversations',
      show_conversations: 'Show conversations',
      rename: 'Rename',
      delete: 'Delete',
      rename_prompt: 'Rename conversation',
      untitled: 'New conversation',
      no_conversations: 'No conversations yet.',
      mode_large: 'Full window',
      mode_docked: 'Dock to the side',
      mode_compact: 'Compact',
      open: 'Open assistant',
      close: 'Close',
      language: 'Language',
      status_checking: 'Checking…',
      status_online: 'Online',
      status_offline: 'Offline',
      status_quota: 'Daily limit reached',
      status_limited: 'Limited mode',
      brand: 'ShopBot',
      mock_badge: 'MOCK DATA',
      recording: 'Recording… speak now',
      cancel: 'Cancel',
      transcribing: 'Transcribing…',
      voice_unsupported: 'Voice input is not supported in this browser.',
      voice_failed: 'Sorry, I could not transcribe that. Please try again or type your question.',
      voice_review: 'Transcribed. Review the text, then send it.',
      mic_denied: 'Microphone access was blocked. Allow it in your browser to use voice input.',
      error_network: 'Sorry, I could not reach the assistant. Please check the connection and try again.',
      error_generic: 'Something went wrong. Please try again.',
      retry: 'Retry',
      copy: 'Copy',
      copied: 'Copied',
      view_results: 'View results',
      open_comparison: 'Open comparison',
      hint: 'ShopBot can make mistakes — please double-check product details.'
    },
    ar: {
      launcher: 'اسأل {assistant}',
      greeting_title: 'مرحباً، أنا {assistant}',
      greeting_text: 'اطلب مني إيجاد منتجات أو المقارنة بين اثنين أو ثلاثة، أو اسأل عن الضمان والإرجاع والتوصيل والدفع عند الاستلام والطلبات. يمكنك أيضاً التحدث بسؤالك.',
      chip_hair_dryer: 'ابحث عن مجفف شعر بأقل من 30 دولاراً',
      chip_hair_brush: 'أرني فرش شعر',
      chip_cod: 'هل يتوفر الدفع عند الاستلام؟',
      chip_order: 'أين طلبي ORD-10025؟',
      placeholder: 'اسأل عن المنتجات أو المقارنات أو السياسات أو الطلبات…',
      send: 'إرسال',
      mic: 'إدخال صوتي',
      new_chat: 'محادثة جديدة',
      hide_conversations: 'إخفاء المحادثات',
      show_conversations: 'إظهار المحادثات',
      rename: 'إعادة تسمية',
      delete: 'حذف',
      rename_prompt: 'إعادة تسمية المحادثة',
      untitled: 'محادثة جديدة',
      no_conversations: 'لا توجد محادثات بعد.',
      mode_large: 'نافذة كاملة',
      mode_docked: 'تثبيت على الجانب',
      mode_compact: 'مصغّرة',
      open: 'فتح المساعد',
      close: 'إغلاق',
      language: 'اللغة',
      status_checking: 'جارٍ التحقق…',
      status_online: 'متصل',
      status_offline: 'غير متصل',
      status_quota: 'تم بلوغ الحد اليومي',
      status_limited: 'الوضع المحدود',
      brand: 'ShopBot',
      mock_badge: 'بيانات تجريبية',
      recording: 'جارٍ التسجيل… تحدث الآن',
      cancel: 'إلغاء',
      transcribing: 'جارٍ التفريغ النصي…',
      voice_unsupported: 'الإدخال الصوتي غير مدعوم في هذا المتصفح.',
      voice_failed: 'عذراً، لم أتمكن من التفريغ النصي. حاول مرة أخرى أو اكتب سؤالك.',
      voice_review: 'تم التفريغ النصي. راجع النص ثم أرسله.',
      mic_denied: 'تم حظر الوصول إلى الميكروفون. اسمح به في المتصفح لاستخدام الإدخال الصوتي.',
      error_network: 'عذراً، لم أتمكن من الوصول إلى المساعد. تحقق من الاتصال وحاول مرة أخرى.',
      error_generic: 'حدث خطأ ما. حاول مرة أخرى.',
      retry: 'إعادة المحاولة',
      copy: 'نسخ',
      copied: 'تم النسخ',
      view_results: 'عرض النتائج',
      open_comparison: 'فتح المقارنة',
      hint: 'قد يخطئ ShopBot — تحقق من تفاصيل المنتج قبل الشراء.'
    }
  };

  /* ---------------------------------------------------------------- helpers */

  function el(tag, opts, children) {
    var node = document.createElement(tag);
    opts = opts || {};
    Object.keys(opts).forEach(function (key) {
      var value = opts[key];
      if (value === null || value === undefined || value === false) { return; }
      if (key === 'class') { node.className = value; }
      else if (key === 'text') { node.textContent = value; }
      else if (key === 'html') { node.innerHTML = value; }
      else if (key === 'dataset') { Object.keys(value).forEach(function (k) { node.dataset[k] = value[k]; }); }
      else if (key.indexOf('on') === 0 && typeof value === 'function') { node.addEventListener(key.slice(2).toLowerCase(), value); }
      else if (value === true) { node.setAttribute(key, ''); }
      else { node.setAttribute(key, value); }
    });
    (children || []).forEach(function (child) {
      if (child === null || child === undefined || child === false) { return; }
      node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    });
    return node;
  }

  function esc(value) {
    return String(value === null || value === undefined ? '' : value).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function renderMarkdown(text) {
    var lines = esc(text || '').split(/\r?\n/);
    var html = '';
    var listOpen = false;
    var paragraph = [];

    function inline(s) {
      return s
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>');
    }
    function closeList() { if (listOpen) { html += '</ul>'; listOpen = false; } }
    function flushParagraph() {
      if (paragraph.length) { html += '<p>' + inline(paragraph.join(' ')) + '</p>'; paragraph = []; }
    }

    lines.forEach(function (line) {
      var trimmed = line.trim();
      if (!trimmed) { closeList(); flushParagraph(); return; }
      var heading = trimmed.match(/^(#{1,4})\s+(.*)$/);
      if (heading) {
        closeList(); flushParagraph();
        var level = Math.min(4, heading[1].length + 1);
        html += '<h' + level + '>' + inline(heading[2]) + '</h' + level + '>';
        return;
      }
      var bullet = trimmed.match(/^[-*]\s+(.*)$/);
      if (bullet) {
        flushParagraph();
        if (!listOpen) { html += '<ul>'; listOpen = true; }
        html += '<li>' + inline(bullet[1]) + '</li>';
        return;
      }
      closeList();
      paragraph.push(trimmed);
    });
    closeList();
    flushParagraph();
    return html || '<p></p>';
  }

  function storageGet(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) { return fallback; }
  }

  function storageSet(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* ignore quota */ }
  }

  function uid() {
    return 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  }

  function isArray(value) {
    return Object.prototype.toString.call(value) === '[object Array]';
  }

  function isMockRequested() {
    var search = location.search || '';
    return /[?&]mock=1(&|$)/.test(search) || /[?&]mock=true(&|$)/.test(search);
  }

  /* ---------------------------------------------------------------- state */

  var settings = {
    apiBase: DEFAULT_API_BASE,
    assistantName: 'ShopBot',
    siteName: '',
    host: {},
    mock: false
  };

  var site = { pages: {}, languages: ['en', 'ar'], currency: 'USD', assistant_name: 'ShopBot', name: '' };

  var state = {
    initialized: false,
    built: false,
    open: false,
    sidebarOpen: true,
    lang: 'en',
    mode: 'large',
    online: null,
    brand: 'ShopBot',
    model: '',
    quota: false,
    limited: false,
    messages: [],
    conversations: [],
    activeId: null,
    sessionId: null,
    busy: false,
    recording: false
  };

  var refs = {};
  var hostEl = null;
  var shadow = null;
  var root = null;
  var mediaRecorder = null;
  var recordedChunks = [];
  var recordTimer = null;
  var healthTimer = null;
  var lastFocused = null;
  var recordingSeconds = 0;

  /* -------------------------------------------------------------- i18n use */

  function t(key, vars) {
    var table = I18N[state.lang] || I18N.en;
    var text = table[key] !== undefined ? table[key] : (I18N.en[key] !== undefined ? I18N.en[key] : key);
    if (vars) {
      Object.keys(vars).forEach(function (name) {
        text = text.replace(new RegExp('\\{' + name + '\\}', 'g'), vars[name]);
      });
    }
    return text;
  }

  /* ----------------------------------------------------------------- build */

  function build() {
    hostEl = document.createElement('div');
    hostEl.setAttribute('data-shopbot-host', '');
    shadow = hostEl.attachShadow({ mode: 'open' });
    shadow.appendChild(el('link', { rel: 'stylesheet', href: CSS_URL }));

    root = el('div', { class: 'sb-root', dir: 'ltr' });
    root.setAttribute('data-mode', state.mode);
    root.setAttribute('data-lang', state.lang);

    /* launcher */
    refs.launcher = el('button', {
      class: 'sb-launcher', type: 'button', 'aria-expanded': 'false',
      onclick: toggle
    }, [
      el('span', { class: 'sb-launcher-icon', text: '💬', 'aria-hidden': 'true' }),
      el('span', { class: 'sb-launcher-label' })
    ]);

    refs.backdrop = el('div', { class: 'sb-backdrop', hidden: true, onclick: close });

    /* sidebar */
    refs.siteLabel = el('span', { class: 'sb-site-label' });
    refs.convList = el('div', { class: 'sb-conv-list' });
    refs.sidebar = el('aside', { class: 'sb-sidebar' }, [
      el('div', { class: 'sb-sidebar-head' }, [
        el('button', { class: 'sb-new-chat', type: 'button', onclick: newConversation }, [
          el('span', { text: '✚', 'aria-hidden': 'true' }),
          el('span', { class: 'sb-new-chat-label' })
        ]),
        el('button', { class: 'sb-icon sb-sidebar-hide', type: 'button', onclick: function () { toggleSidebar(false); } }, [
          el('span', { text: '‹', 'aria-hidden': 'true' })
        ])
      ]),
      refs.convList,
      el('div', { class: 'sb-sidebar-foot' }, [refs.siteLabel])
    ]);

    /* header */
    refs.sidebarToggle = el('button', {
      class: 'sb-icon sb-sidebar-toggle', type: 'button', onclick: function () { toggleSidebar(!state.sidebarOpen); }
    }, [el('span', { text: '☰', 'aria-hidden': 'true' })]);

    refs.name = el('div', { class: 'sb-name' });
    refs.dot = el('span', { class: 'sb-dot' });
    refs.statusText = el('span', { class: 'sb-status-text' });
    refs.system = el('span', { class: 'sb-system' });

    refs.modeButtons = {};
    var modeGroup = el('div', { class: 'sb-modes', role: 'group' });
    [['compact', '▭'], ['docked', '▧'], ['large', '⛶']].forEach(function (pair) {
      var button = el('button', { class: 'sb-icon', type: 'button', onclick: function () { setMode(pair[0]); } }, [
        el('span', { text: pair[1], 'aria-hidden': 'true' })
      ]);
      refs.modeButtons[pair[0]] = button;
      modeGroup.appendChild(button);
    });

    refs.langButton = el('button', { class: 'sb-icon sb-lang-btn', type: 'button', onclick: function () {
      setLang(state.lang === 'ar' ? 'en' : 'ar');
    } });
    refs.closeButton = el('button', { class: 'sb-icon sb-close', type: 'button', onclick: close }, [
      el('span', { text: '✕', 'aria-hidden': 'true' })
    ]);

    var head = el('header', { class: 'sb-head' }, [
      refs.sidebarToggle,
      el('div', { class: 'sb-head-title' }, [
        el('span', { class: 'sb-avatar', text: '🤖', 'aria-hidden': 'true' }),
        el('div', {}, [
          refs.name,
          el('div', { class: 'sb-status' }, [refs.dot, refs.statusText, refs.system])
        ])
      ]),
      el('div', { class: 'sb-head-actions' }, [modeGroup, refs.langButton, refs.closeButton])
    ]);

    /* messages */
    refs.messagesInner = el('div', { class: 'sb-messages-inner' });
    refs.messages = el('div', { class: 'sb-messages', role: 'log', 'aria-live': 'polite' }, [refs.messagesInner]);

    /* recording banner */
    refs.recordingTimer = el('span', { class: 'sb-recording-timer', text: '00:00' });
    refs.recordingLabel = el('span', { class: 'sb-recording-label' });
    refs.recordingCancelText = el('span', { class: 'sb-recording-cancel-text' });
    refs.recording = el('div', { class: 'sb-recording', hidden: true }, [
      el('span', { class: 'sb-recording-dot' }),
      refs.recordingLabel,
      refs.recordingTimer,
      el('button', { class: 'sb-recording-cancel', type: 'button', onclick: cancelRecording }, [refs.recordingCancelText])
    ]);

    refs.transcribing = el('div', { class: 'sb-transcribing', hidden: true });

    /* composer */
    refs.input = el('textarea', { class: 'sb-input', rows: '1' });
    refs.mic = el('button', { class: 'sb-mic', type: 'button', onclick: toggleRecording }, [
      el('span', { text: '🎤', 'aria-hidden': 'true' })
    ]);
    refs.send = el('button', { class: 'sb-send', type: 'submit', disabled: true }, [
      el('span', { text: '➤', 'aria-hidden': 'true' })
    ]);
    refs.hint = el('div', { class: 'sb-composer-hint' });

    var composer = el('form', { class: 'sb-composer', onsubmit: function (event) {
      event.preventDefault();
      send(refs.input.value);
    } }, [
      el('div', { class: 'sb-composer-inner' }, [refs.mic, refs.input, refs.send]),
      refs.hint
    ]);

    refs.main = el('div', { class: 'sb-main' }, [head, refs.messages, refs.recording, composer, refs.transcribing]);

    refs.window = el('section', {
      class: 'sb-window', role: 'dialog', 'aria-modal': 'false', hidden: true
    }, [refs.sidebar, refs.main]);

    refs.toasts = el('div', { class: 'sb-toasts', 'aria-live': 'assertive' });

    root.appendChild(refs.backdrop);
    root.appendChild(refs.window);
    root.appendChild(refs.toasts);
    root.appendChild(refs.launcher);
    shadow.appendChild(root);
    document.body.appendChild(hostEl);

    state.built = true;
    wireStaticEvents();
  }

  /* -------------------------------------------------------------- wiring */

  function wireStaticEvents() {
    refs.input.addEventListener('input', function () {
      refs.input.style.height = 'auto';
      refs.input.style.height = Math.min(refs.input.scrollHeight, 150) + 'px';
      updateSendState();
    });
    refs.input.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        send(refs.input.value);
      }
    });
    shadow.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && state.open) { close(); return; }
      if (event.key === 'Tab' && state.open) { trapFocus(event); }
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && state.open) { close(); }
    });
  }

  /* --------------------------------------------------------- open / close */

  function open() {
    if (!state.built) { build(); }
    state.open = true;
    lastFocused = document.activeElement;
    refs.launcher.setAttribute('aria-expanded', 'true');
    state.sidebarOpen = sidebarShouldOpen();
    applyModeVisibility();
    paint();
    renderConversations();
    renderHealth();
    updateSendState();
    setTimeout(function () { if (refs.input) { refs.input.focus(); } }, 30);
  }

  function close() {
    if (!state.built) { return; }
    state.open = false;
    refs.launcher.setAttribute('aria-expanded', 'false');
    if (state.recording) { cancelRecording(); }
    applyModeVisibility();
    var target = (lastFocused && lastFocused.focus) ? lastFocused : refs.launcher;
    try { target.focus(); } catch (e) { /* ignore */ }
  }

  function toggle() { state.open ? close() : open(); }

  function setMode(mode) {
    state.mode = (mode === 'docked' || mode === 'compact') ? mode : 'large';
    storageSet(STORAGE_MODE, state.mode);
    root.setAttribute('data-mode', state.mode);
    if (state.open) { state.sidebarOpen = sidebarShouldOpen(); }
    applyModeVisibility();
    updateModeButtons();
    scrollDown();
  }

  // The conversation sidebar only has room in the wide (Large) window, and on
  // phones it is an overlay, so it stays closed there (the header toggle can
  // still open it on demand).
  function sidebarShouldOpen() {
    return state.mode === 'large' && window.innerWidth > 640;
  }

  function applyModeVisibility() {
    var isOpen = state.open;
    var isLarge = state.mode === 'large';
    refs.window.hidden = !isOpen;
    refs.window.setAttribute('aria-modal', (isOpen && isLarge) ? 'true' : 'false');
    refs.backdrop.hidden = !(isOpen && isLarge);
    refs.launcher.hidden = isOpen;
    refs.window.classList.toggle('is-sidebar-open', isOpen && state.sidebarOpen);
  }

  function toggleSidebar(openSidebar) {
    state.sidebarOpen = !!openSidebar;
    refs.window.classList.toggle('is-sidebar-open', state.open && state.sidebarOpen);
    if (refs.sidebarToggle) {
      refs.sidebarToggle.setAttribute('aria-label', state.sidebarOpen ? t('hide_conversations') : t('show_conversations'));
    }
  }

  function updateModeButtons() {
    Object.keys(refs.modeButtons).forEach(function (key) {
      refs.modeButtons[key].classList.toggle('is-active', key === state.mode);
    });
  }

  /* ---------------------------------------------------------------- paint */

  function paint() {
    var box = refs.messagesInner;
    box.innerHTML = '';
    if (!state.messages.length) {
      box.appendChild(emptyState());
    } else {
      state.messages.forEach(function (message) { box.appendChild(messageNode(message)); });
    }
    if (state.busy) { box.appendChild(typingNode()); }
    scrollDown();
  }

  function emptyState() {
    var wrap = el('div', { class: 'sb-empty' }, [
      el('div', { class: 'sb-empty-avatar', text: '🤖', 'aria-hidden': 'true' }),
      el('div', { class: 'sb-empty-title', text: t('greeting_title', { assistant: settings.assistantName }) }),
      el('div', { class: 'sb-empty-text', text: t('greeting_text') })
    ]);
    var grid = el('div', { class: 'sb-suggestions' });
    [
      ['💨', 'chip_hair_dryer'],
      ['🪮', 'chip_hair_brush'],
      ['💵', 'chip_cod'],
      ['📦', 'chip_order']
    ].forEach(function (pair) {
      grid.appendChild(el('button', { class: 'sb-suggestion', type: 'button', onclick: function () { send(t(pair[1])); } }, [
        el('span', { class: 'sb-suggestion-icon', text: pair[0], 'aria-hidden': 'true' }),
        el('span', { class: 'sb-suggestion-text', text: t(pair[1]) })
      ]));
    });
    wrap.appendChild(grid);
    return wrap;
  }

  function messageNode(message) {
    var isUser = message.role === 'user';
    var bubble = el('div', { class: 'sb-bubble' + (message.error ? ' is-error' : '') });
    bubble.innerHTML = renderMarkdown(message.text || '');
    if (message.product) { bubble.appendChild(productButton(message.product)); }
    if (message.products && message.products.length > 1) { bubble.appendChild(productChips(message.products)); }
    if (message.search) { bubble.appendChild(actionChip(message.search)); }
    if (message.compare) { bubble.appendChild(compareButton(message.compare)); }

    var children = [bubble];
    if (!isUser && isArray(message.suggestions) && message.suggestions.length) {
      children.push(followupChips(message.suggestions));
    }
    if (!isUser) {
      var tools = el('div', { class: 'sb-msg-tools' });
      tools.appendChild(el('button', { class: 'sb-mini', type: 'button', text: t('copy'), onclick: function () { copyText(message.text); } }));
      if (message.error) {
        tools.appendChild(el('button', { class: 'sb-mini', type: 'button', text: t('retry'), onclick: function () { send(message.retry || ''); } }));
      }
      children.push(tools);
    }
    return el('div', { class: 'sb-msg ' + (isUser ? 'user' : 'bot') }, children);
  }

  function actionChip(action) {
    var label = (action.label ? action.label + ' · ' : '') + t('view_results');
    return el('button', { class: 'sb-action', type: 'button', onclick: function () { go(action); } }, [
      el('span', { text: '🔎', 'aria-hidden': 'true' }),
      el('span', { text: label })
    ]);
  }

  function compareButton(action) {
    return el('button', { class: 'sb-action sb-action-primary', type: 'button', onclick: function () { go(action); } }, [
      el('span', { text: '⚖️', 'aria-hidden': 'true' }),
      el('span', { text: t('open_comparison') })
    ]);
  }

  function productButton(action) {
    var label = (action.label ? action.label + ' • ' : '') + 'View product';
    return el('button', { class: 'sb-action sb-action-primary', type: 'button', onclick: function () { go(action); } }, [
      el('span', { text: '🛍️', 'aria-hidden': 'true' }),
      el('span', { text: label })
    ]);
  }

  function productChips(products) {
    var wrap = el('div', { class: 'sb-product-chips' });
    products.forEach(function (p) {
      var priceStr = p.price ? ' • $' + Number(p.price).toFixed(2) : '';
      var titleShort = (p.title || 'Product').slice(0, 42) + (p.title && p.title.length > 42 ? '…' : '');
      wrap.appendChild(el('button', {
        class: 'sb-action sb-product-chip',
        type: 'button',
        title: p.title || '',
        onclick: function () {
          go({ type: 'navigate', page: 'product', params: { asin: p.parent_asin }, label: p.title });
        }
      }, [
        el('span', { text: '🛍️', 'aria-hidden': 'true' }),
        el('span', { text: titleShort + priceStr })
      ]));
    });
    return wrap;
  }

  function followupChips(suggestions) {
    var wrap = el('div', { class: 'sb-followups' });
    suggestions.slice(0, 3).forEach(function (text) {
      if (!text) { return; }
      wrap.appendChild(el('button', {
        class: 'sb-followup',
        type: 'button',
        text: String(text),
        onclick: function () { send(String(text)); }
      }));
    });
    return wrap;
  }

  function typingNode() {
    return el('div', { class: 'sb-msg bot' }, [
      el('div', { class: 'sb-bubble' }, [
        el('span', { class: 'sb-typing' }, [el('span'), el('span'), el('span')])
      ])
    ]);
  }

  function scrollDown() {
    if (refs.messages) { refs.messages.scrollTop = refs.messages.scrollHeight; }
  }

  function toast(message, kind) {
    if (!refs.toasts) { return; }
    var node = el('div', { class: 'sb-toast' + (kind ? ' is-' + kind : ''), text: message });
    refs.toasts.appendChild(node);
    setTimeout(function () {
      node.style.opacity = '0';
      setTimeout(function () { if (node.parentNode) { node.parentNode.removeChild(node); } }, 300);
    }, 3400);
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text || '').then(function () { toast(t('copied')); }).catch(function () { toast(t('copied')); });
    } else {
      toast(t('copied'));
    }
  }

  /* ----------------------------------------------------------- navigation */

  function go(action) {
    var url = buildUrl(action.page, action.params);
    if (settings.host && typeof settings.host.navigate === 'function') {
      settings.host.navigate(url);
    } else {
      location.hash = url;
    }
    setMode('docked');
  }

  function buildUrl(page, params) {
    params = params || {};
    var template = (site.pages && site.pages[page]) || FALLBACK_PAGES[page] || '#/';
    var hash = template.charAt(0) === '#' ? template.slice(1) : template;
    var qIndex = hash.indexOf('?');
    var path = qIndex >= 0 ? hash.slice(0, qIndex) : hash;
    var query = qIndex >= 0 ? hash.slice(qIndex + 1) : '';

    var seenKeys = {};
    path = path.replace(/\{([a-z_]+)\}/g, function (whole, key) {
      seenKeys[key] = true;
      var value = params[key];
      return (value === null || value === undefined) ? '' : encodeURIComponent(String(value));
    });

    var pairs = [];
    if (query) {
      query.split('&').forEach(function (pair) {
        var key = pair.split('=')[0];
        seenKeys[key] = true;
        var value = params[key];
        if (value === null || value === undefined || value === '') { return; }
        if (isArray(value)) { value = value.join(','); }
        pairs.push(encodeURIComponent(key) + '=' + encodeURIComponent(String(value)));
      });
    }
    Object.keys(params).forEach(function (key) {
      if (seenKeys[key] || params[key] === null || params[key] === undefined || params[key] === '') { return; }
      var value = params[key];
      if (isArray(value)) { value = value.join(','); }
      pairs.push(encodeURIComponent(key) + '=' + encodeURIComponent(String(value)));
    });
    return '#' + path + (pairs.length ? '?' + pairs.join('&') : '');
  }

  /* -------------------------------------------------------------- context */

  function buildContext() {
    var context = null;
    if (settings.host && typeof settings.host.getContext === 'function') {
      try { context = settings.host.getContext(); } catch (e) { context = null; }
    }
    if (!context || typeof context !== 'object') { context = defaultContext(); }
    return context;
  }

  function defaultContext() {
    var hash = location.hash || '#/';
    var path = hash.replace(/^#/, '');
    var qIndex = path.indexOf('?');
    var query = qIndex >= 0 ? path.slice(qIndex + 1) : '';
    path = qIndex >= 0 ? path.slice(0, qIndex) : path;
    var segments = path.split('/').filter(Boolean).map(function (part) { return decodeURIComponent(part); });
    var params = {};
    query.split('&').forEach(function (pair) {
      if (!pair) { return; }
      var kv = pair.split('=');
      params[decodeURIComponent(kv[0] || '')] = decodeURIComponent((kv[1] || '').replace(/\+/g, ' '));
    });

    if (segments[0] === 'product' && segments[1]) { return { page: 'product', asin: segments[1] }; }
    if (segments[0] === 'compare') { return { page: 'compare' }; }
    if (segments[0] === 'cart') { return { page: 'cart' }; }
    if (segments[0] === 'orders') {
      return params.order_id ? { page: 'orders', order_id: params.order_id } : { page: 'orders' };
    }
    if (segments[0] === 'search' || segments[0] === 'category') {
      var context = { page: 'search' };
      if (params.q) { context.query = params.q; }
      var filters = {};
      if (params.category) { filters.category = params.category; }
      if (params.min_price) { filters.min_price = params.min_price; }
      if (params.max_price) { filters.max_price = params.max_price; }
      if (Object.keys(filters).length) { context.filters = filters; }
      return context;
    }
    if (segments[0] === 'help') {
      return segments[1] ? { page: 'help', topic: segments[1] } : { page: 'help' };
    }
    return { page: 'home' };
  }

  /* ----------------------------------------------------------------- send */

  function send(text) {
    text = (text || '').trim();
    if (!text || state.busy) { return; }
    if (!state.built) { build(); }
    open();

    addMessage({ role: 'user', text: text });
    refs.input.value = '';
    refs.input.style.height = 'auto';
    state.busy = true;
    updateSendState();
    paint();

    var payload = { message: text };
    if (state.sessionId) { payload.session_id = state.sessionId; }
    var context = buildContext();
    if (context && Object.keys(context).length) { payload.context = context; }

    if (settings.mock) {
      setTimeout(function () {
        state.busy = false;
        updateSendState();
        handleResponse(mockReply(text));
      }, 320);
      return;
    }

    apiPost('/api/chat', payload).then(function (data) {
      state.busy = false;
      updateSendState();
      if (data && data.session_id) { setSession(data.session_id); }
      handleResponse(data);
    }).catch(function (error) {
      state.busy = false;
      updateSendState();
      var message = (error && error.body && (error.body.text || error.body.error)) || t('error_network');
      addMessage({ role: 'bot', text: message, error: true, retry: text });
      paint();
    });
  }

  function handleResponse(data) {
    if (!data) {
      addMessage({ role: 'bot', text: t('error_generic'), error: true, retry: '' });
      paint();
      return;
    }
    var message = { role: 'bot', text: data.text || '' };
    if (isArray(data.suggestions)) {
      message.suggestions = data.suggestions
        .map(function (item) { return String(item || '').trim(); })
        .filter(Boolean)
        .slice(0, 3);
    }
    if (data.error === 'quota_exhausted' && !state.quota) {
      state.quota = true;
      renderHealth();
    }
    if (data.degraded) {
      state.limited = true;
      state.quota = false;
      renderHealth();
    }
    var actions = isArray(data.actions) ? data.actions : [];
    actions.forEach(function (action) {
      if (!action || action.type !== 'navigate') { return; }
      if (action.page === 'search' && !message.search) { message.search = action; }
      else if (action.page === 'compare' && !message.compare) { message.compare = action; }
      else if (action.page === 'product' && !message.product) { message.product = action; }
    });
    // When products are returned, make sure the dashboard displays the relevant product
    if (isArray(data.products) && data.products.length > 0) {
      var validProducts = data.products.filter(function (p) { return p && p.parent_asin; });
      if (validProducts.length > 0) {
        var topProduct = validProducts[0];
        message.product = {
          type: 'navigate',
          page: 'product',
          params: { asin: topProduct.parent_asin },
          label: topProduct.title || 'View product'
        };
        message.products = validProducts.slice(0, 5);

        if (validProducts.length > 1) {
          var asins = validProducts.map(function (p) { return p.parent_asin; });
          var lastUser = '';
          for (var i = state.messages.length - 1; i >= 0; i--) {
            if (state.messages[i] && state.messages[i].role === 'user' && state.messages[i].text) {
              lastUser = state.messages[i].text;
              break;
            }
          }
          var searchParams = { q: cleanSearchQuery(lastUser), asins: asins.slice(0, 12).join(',') };
          message.search = {
            type: 'navigate',
            page: 'search',
            params: searchParams,
            label: 'All ' + validProducts.length + ' matching products'
          };
        }
      }
    }
    if (!message.text && !message.search && !message.compare && !message.product) {
      message.text = t('error_generic');
      message.error = true;
    }
    addMessage(message);
    paint();
    if (message.product && !message.compare) {
      go(message.product);
    } else if (message.search && !message.compare) {
      go(message.search);
    }
  }

  function addMessage(message) {
    message.time = Date.now();
    state.messages.push(message);
    if (state.messages.length > MAX_MESSAGES) { state.messages = state.messages.slice(-MAX_MESSAGES); }
    persistConversation();
    renderConversations();
  }

  function apiUrl(path) { return settings.apiBase + path; }

  function apiPost(path, payload) {
    return fetch(apiUrl(path), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload || {})
    }).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) {
          var error = new Error((body && (body.text || body.error)) || ('HTTP ' + response.status));
          error.status = response.status;
          error.body = body;
          throw error;
        }
        return body;
      });
    });
  }

  /* --------------------------------------------------------- conversations */

  function loadConversations() {
    var stored = storageGet(STORAGE_CONVS, []);
    state.conversations = isArray(stored) ? stored : [];
    var active = storageGet(STORAGE_ACTIVE, null);
    var exists = state.conversations.some(function (conversation) { return conversation.id === active; });
    if (exists) { state.activeId = active; }
    else if (state.conversations.length) { state.activeId = state.conversations[0].id; }
    else { newConversation(); return; }
    loadActive();
  }

  function activeConversation() {
    var found = null;
    state.conversations.forEach(function (conversation) {
      if (conversation.id === state.activeId) { found = conversation; }
    });
    return found;
  }

  function loadActive() {
    var conversation = activeConversation();
    if (!conversation) { newConversation(); return; }
    state.messages = (conversation.messages || []).slice();
    state.sessionId = conversation.sessionId || null;
    try {
      if (state.sessionId) { sessionStorage.setItem(SESSION_ID, state.sessionId); }
    } catch (e) { /* ignore */ }
  }

  function newConversation() {
    persistConversation();
    var conversation = { id: uid(), title: '', messages: [], sessionId: null, updatedAt: Date.now() };
    state.conversations.unshift(conversation);
    if (state.conversations.length > MAX_CONVERSATIONS) { state.conversations = state.conversations.slice(0, MAX_CONVERSATIONS); }
    state.activeId = conversation.id;
    state.messages = [];
    state.sessionId = null;
    try { sessionStorage.removeItem(SESSION_ID); } catch (e) { /* ignore */ }
    storageSet(STORAGE_CONVS, state.conversations);
    storageSet(STORAGE_ACTIVE, state.activeId);
    if (state.built) { paint(); renderConversations(); }
  }

  function persistConversation() {
    var conversation = activeConversation();
    if (!conversation) { return; }
    conversation.messages = state.messages.slice();
    conversation.sessionId = state.sessionId;
    conversation.updatedAt = Date.now();
    if (!conversation.title) {
      var firstUser = null;
      state.messages.forEach(function (message) { if (!firstUser && message.role === 'user') { firstUser = message; } });
      if (firstUser) { conversation.title = firstUser.text.slice(0, 48); }
    }
    var index = state.conversations.indexOf(conversation);
    if (index > 0) { state.conversations.splice(index, 1); state.conversations.unshift(conversation); }
    storageSet(STORAGE_CONVS, state.conversations);
    storageSet(STORAGE_ACTIVE, state.activeId);
  }

  function setSession(value) {
    state.sessionId = value;
    try { sessionStorage.setItem(SESSION_ID, value); } catch (e) { /* ignore */ }
    persistConversation();
  }

  function switchConversation(id) {
    if (id === state.activeId) { return; }
    persistConversation();
    state.activeId = id;
    storageSet(STORAGE_ACTIVE, state.activeId);
    loadActive();
    paint();
    renderConversations();
  }

  function renameConversation(id) {
    var conversation = null;
    state.conversations.forEach(function (entry) { if (entry.id === id) { conversation = entry; } });
    if (!conversation) { return; }
    var next = window.prompt(t('rename_prompt'), conversation.title || '');
    if (next === null) { return; }
    conversation.title = String(next).slice(0, 60);
    storageSet(STORAGE_CONVS, state.conversations);
    renderConversations();
  }

  function deleteConversation(id) {
    state.conversations = state.conversations.filter(function (conversation) { return conversation.id !== id; });
    if (id === state.activeId) {
      if (state.conversations.length) {
        state.activeId = state.conversations[0].id;
        loadActive();
      } else {
        newConversation();
        return;
      }
    }
    storageSet(STORAGE_CONVS, state.conversations);
    storageSet(STORAGE_ACTIVE, state.activeId);
    if (state.built) { paint(); renderConversations(); }
  }

  function renderConversations() {
    if (!refs.convList) { return; }
    refs.convList.innerHTML = '';
    if (!state.conversations.length) {
      refs.convList.appendChild(el('div', { class: 'sb-conv-empty', text: t('no_conversations') }));
      return;
    }
    state.conversations.forEach(function (conversation) {
      var isActive = conversation.id === state.activeId;
      refs.convList.appendChild(el('div', {
        class: 'sb-conv' + (isActive ? ' is-active' : ''),
        role: 'button',
        tabindex: '0',
        onclick: function () { switchConversation(conversation.id); },
        onkeydown: function (event) { if (event.key === 'Enter') { switchConversation(conversation.id); } }
      }, [
        el('span', { class: 'sb-conv-title', text: conversation.title || t('untitled') }),
        el('span', { class: 'sb-conv-actions' }, [
          el('button', { class: 'sb-conv-btn', type: 'button', title: t('rename'), text: '✎', onclick: function (event) {
            event.stopPropagation();
            renameConversation(conversation.id);
          } }),
          el('button', { class: 'sb-conv-btn', type: 'button', title: t('delete'), text: '🗑', onclick: function (event) {
            event.stopPropagation();
            deleteConversation(conversation.id);
          } })
        ])
      ]));
    });
  }

  /* ------------------------------------------------------------- language */

  function applyLang() {
    state.lang = state.lang === 'ar' ? 'ar' : 'en';
    storageSet(STORAGE_LANG, state.lang);
    root.setAttribute('dir', state.lang === 'ar' ? 'rtl' : 'ltr');
    root.setAttribute('data-lang', state.lang);

    var launcherLabel = refs.launcher.querySelector('.sb-launcher-label');
    if (launcherLabel) { launcherLabel.textContent = t('launcher', { assistant: settings.assistantName }); }
    refs.launcher.setAttribute('aria-label', t('open'));
    var newChatLabel = refs.sidebar.querySelector('.sb-new-chat-label');
    if (newChatLabel) { newChatLabel.textContent = t('new_chat'); }
    refs.sidebarToggle.setAttribute('aria-label', state.sidebarOpen ? t('hide_conversations') : t('show_conversations'));
    refs.siteLabel.textContent = settings.siteName || '';
    refs.name.textContent = settings.assistantName;
    refs.recordingLabel.textContent = t('recording');
    refs.recordingCancelText.textContent = t('cancel');
    refs.transcribing.textContent = t('transcribing');
    refs.input.setAttribute('placeholder', t('placeholder'));
    refs.input.setAttribute('aria-label', t('placeholder'));
    refs.mic.setAttribute('aria-label', t('mic'));
    refs.mic.setAttribute('title', t('mic'));
    refs.send.setAttribute('aria-label', t('send'));
    refs.hint.textContent = t('hint');
    refs.langButton.textContent = state.lang === 'ar' ? 'EN' : 'ع';
    refs.langButton.setAttribute('aria-label', t('language'));
    refs.closeButton.setAttribute('aria-label', t('close'));
    updateModeButtons();

    if (state.built) { paint(); renderConversations(); renderHealth(); }
    try { window.dispatchEvent(new CustomEvent('shopbot:lang', { detail: state.lang })); } catch (e) { /* ignore */ }
  }

  function setLang(lang) {
    state.lang = lang === 'ar' ? 'ar' : 'en';
    applyLang();
  }

  function getLang() { return state.lang; }

  /* --------------------------------------------------------------- health */

  function setHealth(health) {
    state.online = health.online;
    state.brand = health.brand || 'ShopBot';
    state.model = health.model || '';
    state.quota = !!health.quota;
    state.limited = !!health.limited;
    renderHealth();
  }

  function renderHealth() {
    if (!refs.statusText) { return; }
    refs.system.innerHTML = '';
    if (state.online === null) {
      refs.statusText.textContent = t('status_checking');
      refs.dot.classList.remove('is-online', 'is-offline', 'is-quota', 'is-limited');
      return;
    }
    if (state.online === false) {
      refs.statusText.textContent = t('status_offline');
      refs.dot.classList.remove('is-online', 'is-quota', 'is-limited');
      refs.dot.classList.add('is-offline');
      return;
    }
    refs.dot.classList.remove('is-offline');
    if (state.limited) {
      refs.statusText.textContent = t('status_limited');
      refs.dot.classList.remove('is-online', 'is-quota');
      refs.dot.classList.add('is-limited');
      refs.system.appendChild(el('span', { class: 'sb-badge is-limited', text: t('status_limited') }));
      return;
    }
    refs.dot.classList.remove('is-limited');
    if (state.quota) {
      refs.statusText.textContent = t('status_quota');
      refs.dot.classList.remove('is-online');
      refs.dot.classList.add('is-quota');
      return;
    }
    refs.statusText.textContent = t('status_online');
    refs.dot.classList.remove('is-quota');
    refs.dot.classList.add('is-online');
    if (settings.mock) {
      refs.system.appendChild(el('span', { class: 'sb-badge is-mock', text: t('mock_badge') }));
      return;
    }
    refs.system.appendChild(el('span', { class: 'sb-badge', text: state.brand }));
    if (state.model) { refs.system.appendChild(el('span', { text: '· ' + state.model })); }
  }

  function checkHealth() {
    if (settings.mock) {
      setHealth({ online: true, brand: 'ShopBot', model: 'mock' });
      return;
    }
    fetch(apiUrl('/api/health')).then(function (response) {
      return response.ok ? response.json() : {};
    }).then(function (data) {
      setHealth({
        online: true,
        brand: (data && data.name) || 'ShopBot',
        model: (data && data.model) || '',
        quota: !!(data && data.llm === 'quota_exhausted'),
        limited: !!(data && data.llm === 'limited')
      });
    }).catch(function () {
      setHealth({ online: false, brand: 'ShopBot', model: '', quota: false, limited: false });
    });
  }

  function loadSite() {
    if (settings.mock) {
      site.pages = FALLBACK_PAGES;
      applySite();
      return;
    }
    fetch(apiUrl('/api/site')).then(function (response) {
      return response.ok ? response.json() : null;
    }).then(function (data) {
      if (data && !data.error) { site = data; }
      if (!site.pages || !Object.keys(site.pages).length) { site.pages = FALLBACK_PAGES; }
      applySite();
    }).catch(function () {
      site.pages = FALLBACK_PAGES;
      applySite();
    });
  }

  function applySite() {
    if (site.assistant_name) { settings.assistantName = site.assistant_name; }
    if (site.name) { settings.siteName = site.name; }
    if (isArray(site.languages) && site.languages.length && site.languages.indexOf(state.lang) < 0) {
      state.lang = site.languages[0] === 'ar' ? 'ar' : 'en';
    }
    applyLang();
  }

  /* ---------------------------------------------------------------- voice */

  function toggleRecording() {
    if (state.recording) { stopRecording(); return; }
    if (!navigator.mediaDevices || !window.MediaRecorder) { toast(t('voice_unsupported'), 'warn'); return; }
    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stream) {
      mediaRecorder = new MediaRecorder(stream);
      recordedChunks = [];
      mediaRecorder.ondataavailable = function (event) { if (event.data && event.data.size) { recordedChunks.push(event.data); } };
      mediaRecorder.onstop = function () {
        stream.getTracks().forEach(function (track) { track.stop(); });
        uploadRecording();
      };
      mediaRecorder.start();
      startRecordingUI();
    }).catch(function () { toast(t('mic_denied'), 'warn'); });
  }

  function startRecordingUI() {
    state.recording = true;
    recordingSeconds = 0;
    refs.mic.classList.add('is-recording');
    refs.recording.hidden = false;
    refs.recordingTimer.textContent = '00:00';
    recordTimer = setInterval(function () {
      recordingSeconds += 1;
      var mm = String(Math.floor(recordingSeconds / 60)).padStart(2, '0');
      var ss = String(recordingSeconds % 60).padStart(2, '0');
      refs.recordingTimer.textContent = mm + ':' + ss;
    }, 1000);
  }

  function stopRecordingUI() {
    state.recording = false;
    if (recordTimer) { clearInterval(recordTimer); recordTimer = null; }
    refs.mic.classList.remove('is-recording');
    refs.recording.hidden = true;
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') { mediaRecorder.stop(); }
    stopRecordingUI();
  }

  function cancelRecording() {
    if (mediaRecorder) { mediaRecorder.onstop = null; }
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      try { mediaRecorder.stop(); } catch (e) { /* ignore */ }
    }
    recordedChunks = [];
    stopRecordingUI();
  }

  function uploadRecording() {
    var blob = new Blob(recordedChunks, { type: 'audio/webm' });
    if (!blob.size) { return; }
    refs.transcribing.hidden = false;
    var form = new FormData();
    form.append('audio', blob, 'recording.webm');
    fetch(apiUrl('/api/transcribe'), { method: 'POST', body: form })
      .then(function (response) { return response.json().then(function (body) { return { ok: response.ok, body: body }; }); })
      .then(function (result) {
        refs.transcribing.hidden = true;
        if (!result.ok || !result.body || !result.body.text) { toast(t('voice_failed'), 'warn'); return; }
        refs.input.value = result.body.text;
        refs.input.style.height = 'auto';
        refs.input.style.height = Math.min(refs.input.scrollHeight, 150) + 'px';
        updateSendState();
        toast(t('voice_review'));
        refs.input.focus();
      })
      .catch(function () { refs.transcribing.hidden = true; toast(t('voice_failed'), 'warn'); });
  }

  /* --------------------------------------------------------------- mock */

  function mockReply(text) {
    var lower = String(text || '').toLowerCase();
    var orderMatch = lower.match(/ord-\d+/);
    if (orderMatch) {
      return {
        text: 'Order **' + orderMatch[0].toUpperCase() + '** is in transit and expected within 2 days. (mock data)',
        actions: [{ type: 'navigate', page: 'orders', params: { order_id: orderMatch[0].toUpperCase() } }]
      };
    }
    if (lower.indexOf('compare') >= 0 || lower.indexOf(' vs ') >= 0) {
      return {
        text: 'Here is a short comparison of the two products. (mock data)',
        actions: [{ type: 'navigate', page: 'compare', params: { asins: ['B085BB7B1M', 'B09X9BG4FC'] } }]
      };
    }
    if (lower.indexOf('return') >= 0 || lower.indexOf('warranty') >= 0 || lower.indexOf('delivery') >= 0 || lower.indexOf('cash') >= 0 || lower.indexOf('shipping') >= 0) {
      return { text: 'Here is what our store policy says about that. (mock data)' };
    }
    return {
      text: 'I found a few products that match. (mock data)',
      actions: [{ type: 'navigate', page: 'search', params: { q: 'hair brush', max_price: 15 }, label: 'Hair brush under $15' }],
      suggestions: ['Tell me more about the first one', 'Compare the first two', 'Show me cheaper options']
    };
  }

  /* --------------------------------------------------------------- focus */

  function focusable() {
    var nodes = root.querySelectorAll('button:not([disabled]), [href], input, textarea, select, [tabindex]:not([tabindex="-1"])');
    return Array.prototype.filter.call(nodes, function (node) {
      if (node.closest('[hidden]')) { return false; }
      return !!(node.offsetWidth || node.offsetHeight || node.getClientRects().length);
    });
  }

  function trapFocus(event) {
    var nodes = focusable();
    if (!nodes.length) { return; }
    var first = nodes[0];
    var last = nodes[nodes.length - 1];
    var active = shadow.activeElement;
    if (event.shiftKey && (active === first || !root.contains(active))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  }

  /* ------------------------------------------------------------ send state */

  function updateSendState() {
    if (!refs.send) { return; }
    var hasText = !!(refs.input && refs.input.value.trim());
    refs.send.disabled = state.busy || !hasText;
  }

  /* ----------------------------------------------------------------- init */

  function init(options) {
    options = options || {};
    if (options.apiBase) { settings.apiBase = String(options.apiBase).replace(/\/+$/, ''); }
    if (options.host && typeof options.host === 'object') {
      settings.host = {
        navigate: typeof options.host.navigate === 'function' ? options.host.navigate : settings.host.navigate,
        getContext: typeof options.host.getContext === 'function' ? options.host.getContext : settings.host.getContext
      };
    }
    if (options.assistantName) { settings.assistantName = options.assistantName; }
    if (typeof options.mock === 'boolean') { settings.mock = options.mock; }

    if (!state.initialized) {
      state.initialized = true;
      settings.mock = settings.mock || isMockRequested();
      var savedLang = storageGet(STORAGE_LANG, null);
      state.lang = savedLang === 'ar' ? 'ar' : 'en';
      var savedMode = storageGet(STORAGE_MODE, null);
      state.mode = (savedMode === 'docked' || savedMode === 'compact') ? savedMode : 'large';
      build();
      loadConversations();
      applyLang();
      updateModeButtons();
      loadSite();
      checkHealth();
      if (healthTimer) { clearInterval(healthTimer); }
      healthTimer = setInterval(function () { checkHealth(); }, HEALTH_INTERVAL);
      return;
    }
    if (state.built) { applyLang(); }
  }

  /* --------------------------------------------------------------- export */

  var api = {
    init: init,
    open: open,
    close: close,
    toggle: toggle,
    send: send,
    reset: newConversation,
    setLang: setLang,
    getLang: getLang,
    setMode: setMode,
    getMode: function () { return state.mode; },
    isOpen: function () { return state.open; }
  };

  window.ShopBot = Object.assign(window.ShopBot || {}, api);

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () { init(); });
  } else {
    init();
  }
})();
