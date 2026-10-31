/* ==========================================================================
   ShopStore — shared frontend utilities
   --------------------------------------------------------------------------
   Vanilla JS. No framework, no build step, no third-party runtime deps.

   Exposes one global, `window.ShopStore` (also aliased as `SS`), used by the
   storefront SPA (app.js). The ShopBot widget (shopbot/shopbot.js) is
   Shadow-DOM isolated and does not depend on it:

     * DOM helpers            $ · el · esc
     * formatting             formatUsd · formatAed · hasPrice · starsText · initialOf
     * markdown               renderMarkdown
     * toasts                 toast
     * HTTP                   apiGet · apiPost  (JSON in, JSON out)
     * routing                navigate
     * cart / compare lists   cart · compare  (persisted in localStorage)
     * a tiny event bus        on · emit

   The API base is derived from the page origin, so the same files work when
   served by the ShopBot app (app.py, port 5000). The literal localhost
   fallback is only used for file:// development.
   ========================================================================== */

(function () {
  'use strict';

  var LOCAL_API_BASE = 'http://localhost:5000'; // file:// fallback (dev only)

  var API_BASE = (window.location.protocol === 'http:' || window.location.protocol === 'https:')
    ? window.location.origin
    : LOCAL_API_BASE;

  var USD_TO_AED = 3.67;   // fixed conversion rate from the project spec
  var MAX_COMPARE = 3;     // products that can be compared side by side

  var STORAGE_CART = 'shopstore_cart';
  var STORAGE_COMPARE = 'shopstore_compare';

  /* ---------------------------------------------------------------- DOM */

  function $(id) { return document.getElementById(id); }

  function el(tag, opts, children) {
    var node = document.createElement(tag);
    opts = opts || {};
    Object.keys(opts).forEach(function (key) {
      var value = opts[key];
      if (value === null || value === undefined) { return; }
      if (key === 'class') { node.className = value; }
      else if (key === 'text') { node.textContent = value; }
      else if (key === 'html') { node.innerHTML = value; }
      else if (key === 'dataset') { Object.keys(value).forEach(function (k) { node.dataset[k] = value[k]; }); }
      else if (key.indexOf('on') === 0 && typeof value === 'function') { node.addEventListener(key.slice(2).toLowerCase(), value); }
      else if (value === true) { node.setAttribute(key, ''); }
      else { node.setAttribute(key, value); }
    });
    (children || []).forEach(function (child) {
      if (child === null || child === undefined) { return; }
      node.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
    });
    return node;
  }

  function esc(value) {
    return String(value === null || value === undefined ? '' : value).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  /* ------------------------------------------------------------ format */

  function formatUsd(price) { return '$' + Number(price).toFixed(2); }
  function formatAed(price) { return 'AED ' + (Number(price) * USD_TO_AED).toFixed(2); }
  function hasPrice(price) { return price !== null && price !== undefined && Number(price) > 0; }

  function starsText(rating) {
    var r = Math.max(0, Math.min(5, Math.round(Number(rating) || 0)));
    return '★★★★★'.slice(0, r) + '☆☆☆☆☆'.slice(0, 5 - r);
  }

  function initialOf(title) {
    var match = String(title || '').trim().match(/[A-Za-z0-9\u0600-\u06FF]/);
    return match ? match[0].toUpperCase() : '?';
  }

  /* small markdown renderer (bold, italics, inline code, bullets, headings) */
  function renderMarkdown(text) {
    var escaped = esc(text || '');
    var lines = escaped.split(/\r?\n/);
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
      var bullet = trimmed.match(/^[-*•]\s+(.*)$/);
      if (heading) {
        closeList(); flushParagraph();
        var level = Math.min(4, heading[1].length + 2);
        html += '<h' + level + '>' + inline(heading[2]) + '</h' + level + '>';
      } else if (bullet) {
        flushParagraph();
        if (!listOpen) { html += '<ul>'; listOpen = true; }
        html += '<li>' + inline(bullet[1]) + '</li>';
      } else {
        closeList();
        paragraph.push(trimmed);
      }
    });
    closeList();
    flushParagraph();
    return html || '<p></p>';
  }

  /* ------------------------------------------------------------ toasts */

  function toast(message, kind) {
    var container = $('toastContainer');
    if (!container) { return; }
    var node = el('div', { class: 'toast' + (kind ? ' is-' + kind : ''), text: message });
    container.appendChild(node);
    setTimeout(function () {
      node.style.opacity = '0';
      setTimeout(function () { if (node.parentNode) { node.parentNode.removeChild(node); } }, 300);
    }, 3600);
  }

  /* -------------------------------------------------------------- HTTP */

  function apiGet(path) {
    return fetch(API_BASE + path).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) {
          var error = new Error((body && body.error) || ('HTTP ' + response.status));
          error.status = response.status; error.body = body; throw error;
        }
        return body;
      });
    });
  }

  function apiPost(path, payload) {
    return fetch(API_BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload || {})
    }).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) {
          var error = new Error((body && (body.text || body.error)) || ('HTTP ' + response.status));
          error.status = response.status; error.body = body; throw error;
        }
        return body;
      });
    });
  }

  /* ----------------------------------------------------------- routing */

  function navigate(hash) {
    var target = hash || '#/';
    if (window.location.hash === target) {
      emit('route', target); // same route: force a re-render
    } else {
      window.location.hash = target;
      emit('route', target);
    }
  }

  /* ------------------------------------------------------------- event bus */

  var listeners = {};

  function on(name, handler) {
    (listeners[name] = listeners[name] || []).push(handler);
  }

  function emit(name, detail) {
    (listeners[name] || []).forEach(function (handler) {
      try { handler(detail); } catch (e) { /* ignore listener errors */ }
    });
  }

  /* ------------------------------------------------------------ storage */

  function readJson(key, fallback) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) { return fallback; }
  }

  function writeJson(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* ignore quota */ }
  }

  /* -------------------------------------------------------------- cart */

  function getCart() {
    var items = readJson(STORAGE_CART, []);
    return Array.isArray(items) ? items : [];
  }

  function saveCart(items) {
    writeJson(STORAGE_CART, items);
    emit('cart', cartCount());
  }

  function cartCount() {
    return getCart().reduce(function (sum, item) { return sum + (Number(item.qty) || 0); }, 0);
  }

  function cartAdd(item, qty) {
    var items = getCart();
    var index = -1;
    items.forEach(function (entry, i) { if (entry.parent_asin === item.parent_asin) { index = i; } });
    var amount = Math.max(1, Number(qty) || 1);
    if (index >= 0) {
      items[index].qty = (Number(items[index].qty) || 0) + amount;
    } else {
      items.push({
        parent_asin: item.parent_asin, title: item.title || '',
        price: hasPrice(item.price) ? item.price : null,
        image_url: item.image_url || '', qty: amount
      });
    }
    saveCart(items);
  }

  function cartRemove(asin) {
    saveCart(getCart().filter(function (item) { return item.parent_asin !== asin; }));
  }

  function cartSetQty(asin, qty) {
    var amount = Math.max(0, Number(qty) || 0);
    var items = getCart();
    if (amount === 0) {
      saveCart(items.filter(function (item) { return item.parent_asin !== asin; }));
      return;
    }
    items.forEach(function (item) { if (item.parent_asin === asin) { item.qty = amount; } });
    saveCart(items);
  }

  function cartTotal() {
    return getCart().reduce(function (sum, item) {
      return sum + (hasPrice(item.price) ? Number(item.price) * (Number(item.qty) || 0) : 0);
    }, 0);
  }

  function cartClear() { saveCart([]); }

  /* ----------------------------------------------------------- compare */

  function getCompare() {
    var asins = readJson(STORAGE_COMPARE, []);
    return Array.isArray(asins) ? asins.slice(0, MAX_COMPARE) : [];
  }

  function saveCompare(asins) {
    writeJson(STORAGE_COMPARE, asins.slice(0, MAX_COMPARE));
    emit('compare', asins.length);
  }

  function inCompare(asin) { return getCompare().indexOf(asin) >= 0; }

  function toggleCompare(asin) {
    var asins = getCompare();
    var index = asins.indexOf(asin);
    if (index >= 0) { asins.splice(index, 1); }
    else {
      if (asins.length >= MAX_COMPARE) { return false; }
      asins.push(asin);
    }
    saveCompare(asins);
    return true;
  }

  function clearCompare() { saveCompare([]); }

  /* ------------------------------------------------------------ exports */

  window.ShopStore = {
    API_BASE: API_BASE,
    USD_TO_AED: USD_TO_AED,
    MAX_COMPARE: MAX_COMPARE,
    $: $, el: el, esc: esc,
    formatUsd: formatUsd, formatAed: formatAed, hasPrice: hasPrice,
    starsText: starsText, initialOf: initialOf, renderMarkdown: renderMarkdown,
    toast: toast, apiGet: apiGet, apiPost: apiPost, navigate: navigate,
    on: on, emit: emit,
    cart: {
      all: getCart, count: cartCount, add: cartAdd, remove: cartRemove,
      setQty: cartSetQty, total: cartTotal, clear: cartClear
    },
    compare: {
      all: getCompare, has: inCompare, toggle: toggleCompare, clear: clearCompare
    }
  };

  window.SS = window.ShopStore;
})();
