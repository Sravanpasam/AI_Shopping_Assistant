/* ==========================================================================
   ShopStore — storefront single-page app (hash router)
   --------------------------------------------------------------------------
   Vanilla JS. No framework, no build step, no third-party runtime deps.

   Routes (must match sites/<site>/site.json "pages"):
     #/                                   home
     #/search?q=&min_price=&max_price=&category=&sort=&page=
     #/category/{category}
     #/product/{asin}
     #/compare?asins=a,b,c
     #/cart
     #/orders?order_id=ORD-10025
     #/help/{topic}                        returns|warranty|shipping|payment|orders

   Data comes from the shared storefront API:
     GET  /api/products   GET /api/products/<asin>   POST /api/compare
     GET  /api/orders/<order_id>   GET /api/help/<topic>   (added in STEP 2)

   The ShopBot assistant is a self-contained Shadow-DOM widget (shopbot/) that
   mounts itself; this file only wires it up here — giving it the store's
   navigate() and a page-context builder — and drives its language toggle.
   ========================================================================== */

(function () {
  'use strict';

  var SS = window.ShopStore;
  var el = SS.el, $ = SS.$;

  /* ---------------------------------------------------------------- config */

  var PAGE_SIZE = 24;

  var CATEGORY_ORDER = [
    'hair', 'skin care', 'makeup', 'fragrance',
    'bath and body', 'nails', 'tools and accessories', 'other'
  ];
  var CATEGORY_LABELS = {
    'hair': 'Hair',
    'skin care': 'Skin care',
    'makeup': 'Makeup',
    'fragrance': 'Fragrance',
    'bath and body': 'Bath & body',
    'nails': 'Nails',
    'tools and accessories': 'Tools & accessories',
    'other': 'Other'
  };

  var SORT_OPTIONS = [
    ['relevance', 'Relevance'],
    ['price_asc', 'Price: low to high'],
    ['price_desc', 'Price: high to low'],
    ['rating', 'Top rated'],
    ['reviews', 'Most reviewed']
  ];

  var HELP_TOPICS = ['returns', 'warranty', 'shipping', 'payment', 'orders'];
  var HELP_LABELS = {
    returns: 'Returns & refunds',
    warranty: 'Warranty',
    shipping: 'Shipping & delivery',
    payment: 'Payment & cash on delivery',
    orders: 'Order tracking'
  };

  /* --------------------------------------------------------------- helpers */

  function categoryLabel(value) { return CATEGORY_LABELS[value] || value; }
  function hasPrice(price) { return SS.hasPrice(price); }
  function formatUsd(price) { return SS.formatUsd(price); }
  function formatAed(price) { return SS.formatAed(price); }
  function toast(message, kind) { SS.toast(message, kind); }

  function parseQuery(text) {
    var out = {};
    (text || '').split('&').forEach(function (pair) {
      if (!pair) { return; }
      var kv = pair.split('=');
      var key = decodeURIComponent((kv[0] || '').replace(/\+/g, ' '));
      out[key] = decodeURIComponent((kv[1] || '').replace(/\+/g, ' '));
    });
    return out;
  }

  function encodeQuery(params) {
    return Object.keys(params).filter(function (key) {
      var value = params[key];
      return value !== null && value !== undefined && value !== '';
    }).map(function (key) {
      return encodeURIComponent(key) + '=' + encodeURIComponent(params[key]);
    }).join('&');
  }

  function parseRoute() {
    var hash = window.location.hash || '#/';
    var queryIndex = hash.indexOf('?');
    var path = queryIndex >= 0 ? hash.slice(0, queryIndex) : hash;
    var params = parseQuery(queryIndex >= 0 ? hash.slice(queryIndex + 1) : '');
    path = path.replace(/^#/, '');
    var segments = path.split('/').filter(function (part) { return part.length; })
      .map(function (part) { return decodeURIComponent(part); });

    var name = 'home';
    if (segments[0] === 'search') { name = 'search'; }
    else if (segments[0] === 'category') { name = 'category'; }
    else if (segments[0] === 'product') { name = 'product'; }
    else if (segments[0] === 'compare') { name = 'compare'; }
    else if (segments[0] === 'cart') { name = 'cart'; }
    else if (segments[0] === 'orders') { name = 'orders'; }
    else if (segments[0] === 'help') { name = 'help'; }
    else if (segments.length) { name = 'notfound'; }

    return { name: name, segments: segments, params: params };
  }

  function searchHash(overrides) {
    var query = encodeQuery({
      q: overrides && overrides.q,
      asins: overrides && overrides.asins,
      category: overrides && overrides.category,
      min_price: overrides && overrides.min_price,
      max_price: overrides && overrides.max_price,
      sort: overrides && overrides.sort,
      page: overrides && overrides.page
    });
    return '#/search' + (query ? '?' + query : '');
  }

  function compareHash(asins) {
    return '#/compare?asins=' + encodeURIComponent((asins || []).join(','));
  }

  // Page context handed to the assistant on every turn (System B uses it to
  // resolve "this" / "it"; System A ignores it). Built from the current route.
  function assistantContext() {
    var route = parseRoute();
    if (route.name === 'product') { return { page: 'product', asin: route.segments[1] }; }
    if (route.name === 'compare') { return { page: 'compare' }; }
    if (route.name === 'cart') { return { page: 'cart', cart_count: SS.cart.count() }; }
    if (route.name === 'orders') {
      return route.params.order_id ? { page: 'orders', order_id: route.params.order_id } : { page: 'orders' };
    }
    if (route.name === 'search' || route.name === 'category') {
      var context = { page: 'search' };
      if (route.params.q) { context.query = route.params.q; }
      var filters = {};
      if (route.params.category) { filters.category = route.params.category; }
      if (route.params.min_price) { filters.min_price = route.params.min_price; }
      if (route.params.max_price) { filters.max_price = route.params.max_price; }
      if (Object.keys(filters).length) { context.filters = filters; }
      return context;
    }
    if (route.name === 'help') {
      return route.segments[1] ? { page: 'help', topic: route.segments[1] } : { page: 'help' };
    }
    return { page: 'home' };
  }

  function mediaNode(item, extraClass) {
    var media = el('div', { class: 'card-media' + (extraClass ? ' ' + extraClass : '') });
    if (item.image_url) {
      media.appendChild(el('img', { src: item.image_url, alt: item.title || '', loading: 'lazy' }));
    } else {
      media.appendChild(el('div', { class: 'card-media-placeholder' }, [
        el('span', { class: 'card-media-placeholder-icon', text: '🛍️' }),
        el('span', { class: 'card-media-placeholder-initial', text: SS.initialOf(item.title) })
      ]));
    }
    return media;
  }

  function starsRow(rating, count) {
    var children = [
      el('span', { class: 'stars', text: SS.starsText(rating) }),
      el('span', { class: 'rating-value', text: (Number(rating) || 0).toFixed(1) })
    ];
    if (count !== undefined && count !== null) {
      children.push(el('span', { class: 'rating-count', text: '(' + (count || 0) + ')' }));
    }
    return el('div', { class: 'card-rating' }, children);
  }

  function priceNode(price, className) {
    if (!hasPrice(price)) {
      return el('div', { class: 'price-unavailable ' + (className || ''), text: 'Price unavailable' });
    }
    return el('div', { class: 'price-block ' + (className || '') }, [
      el('span', { class: 'price-usd', text: formatUsd(price) }),
      el('span', { class: 'price-aed', text: formatAed(price) })
    ]);
  }

  function button(label, className, onClick, attrs) {
    var opts = { class: className, type: 'button', text: label, onclick: onClick };
    Object.keys(attrs || {}).forEach(function (k) { opts[k] = attrs[k]; });
    return el('button', opts);
  }

  function link(label, href, className) {
    return el('a', { class: className, href: href, text: label });
  }

  function pageHead(title, subtitle) {
    return el('div', { class: 'page-head' }, [
      el('h1', { class: 'page-title', text: title }),
      subtitle ? el('p', { class: 'page-sub', text: subtitle }) : null
    ]);
  }

  function statusNote(text, isError) {
    return el('p', { class: 'page-note' + (isError ? ' is-error' : ''), text: text });
  }

  /* ------------------------------------------------------------ product card */

  function productCard(item) {
    var asin = item.parent_asin;
    var inCompare = SS.compare.has(asin);

    var media = mediaNode(item);
    media.appendChild(button(
      inCompare ? '✓ In compare' : '+ Compare',
      'card-compare-btn' + (inCompare ? ' is-on' : ''),
      function () {
        if (!SS.compare.toggle(asin)) { toast('You can compare up to 3 products.'); return; }
        render();
      }
    ));

    return el('article', { class: 'product-card', dataset: { asin: asin } }, [
      media,
      el('div', { class: 'card-body' }, [
        el('a', { class: 'card-title', href: '#/product/' + encodeURIComponent(asin), text: item.title || 'N/A' }),
        item.store ? el('div', { class: 'card-store', text: item.store }) : null,
        starsRow(item.average_rating, item.rating_number),
        el('div', { class: 'card-category-tag', text: categoryLabel(item.category) }),
        priceNode(item.price, 'card-price'),
        el('div', { class: 'card-actions' }, [
          button('Add to cart', 'btn btn-primary card-cart-btn', function () {
            SS.cart.add(item, 1);
            toast('Added to cart');
          }),
          link('Details', '#/product/' + encodeURIComponent(asin), 'btn btn-ghost card-details-link')
        ])
      ])
    ]);
  }

  function productGrid(items) {
    var grid = el('div', { class: 'product-grid' });
    items.forEach(function (item) { grid.appendChild(productCard(item)); });
    return grid;
  }

  function skeletonGrid(count) {
    var grid = el('div', { class: 'product-grid' });
    for (var i = 0; i < (count || 8); i++) {
      grid.appendChild(el('div', { class: 'skeleton-card' }, [
        el('div', { class: 'skeleton-media' }),
        el('div', { class: 'skeleton-line' }),
        el('div', { class: 'skeleton-line short' })
      ]));
    }
    return grid;
  }

  function pagination(page, pages, buildHash) {
    if (pages <= 1) { return null; }
    var wrap = el('nav', { class: 'pagination', 'aria-label': 'Pagination' });
    var prev = button('← Previous', 'page-btn', function () { SS.navigate(buildHash(page - 1)); });
    prev.disabled = page <= 1;
    wrap.appendChild(prev);

    var start = Math.max(1, page - 2);
    var end = Math.min(pages, start + 4);
    start = Math.max(1, end - 4);
    for (var p = start; p <= end; p++) {
      (function (current) {
        wrap.appendChild(button(String(current), 'page-btn' + (current === page ? ' is-current' : ''), function () {
          SS.navigate(buildHash(current));
        }));
      })(p);
    }

    var next = button('Next →', 'page-btn', function () { SS.navigate(buildHash(page + 1)); });
    next.disabled = page >= pages;
    wrap.appendChild(next);
    return wrap;
  }

  /* ------------------------------------------------------------------ home */

  function renderHome(view) {
    view.appendChild(el('section', { class: 'hero' }, [
      el('h1', { class: 'hero-title', text: 'Beauty, care and everyday essentials' }),
      el('p', { class: 'hero-sub', text: 'Browse the catalogue, compare products side by side, and ask ShopBot for help in your own words.' }),
      el('div', { class: 'hero-actions' }, [
        link('Browse all products', searchHash({}), 'btn btn-primary'),
        button('Ask ShopBot', 'btn btn-ghost', function () { window.ShopBot.open(); })
      ])
    ]));

    view.appendChild(el('div', { class: 'promo-banner' }, [
      el('span', { text: 'Free standard delivery on orders of AED 150 or more' }),
      el('span', { class: 'promo-dot', text: '•' }),
      el('span', { text: '14-day returns on most items' }),
      el('span', { class: 'promo-dot', text: '•' }),
      el('span', { text: '12-month warranty on electronics' })
    ]));

    var tiles = el('div', { class: 'tile-grid', id: 'homeTiles' });
    view.appendChild(el('section', { class: 'page-section' }, [
      el('h2', { class: 'section-title', text: 'Shop by category' }), tiles
    ]));

    var topRated = el('div', { class: 'home-row' }, [skeletonGrid(6)]);
    var mostReviewed = el('div', { class: 'home-row' }, [skeletonGrid(6)]);
    view.appendChild(el('section', { class: 'page-section' }, [
      el('h2', { class: 'section-title', text: 'Top rated' }), topRated
    ]));
    view.appendChild(el('section', { class: 'page-section' }, [
      el('h2', { class: 'section-title', text: 'Most reviewed' }), mostReviewed
    ]));

    SS.apiGet('/api/products?page_size=1').then(function (data) {
      var tilesParent = $('homeTiles');
      if (!tilesParent) { return; }
      tilesParent.innerHTML = '';
      var counts = {};
      ((data.facets && data.facets.categories) || []).forEach(function (f) { counts[f.value] = f.count; });
      CATEGORY_ORDER.forEach(function (name) {
        tilesParent.appendChild(el('a', { class: 'tile', href: '#/category/' + encodeURIComponent(name) }, [
          el('span', { class: 'tile-name', text: categoryLabel(name) }),
          el('span', { class: 'tile-count', text: (counts[name] || 0) + ' products' })
        ]));
      });
    }).catch(function () {
      var tilesParent = $('homeTiles');
      if (tilesParent) { tilesParent.innerHTML = ''; tilesParent.appendChild(statusNote('Categories are unavailable right now.', true)); }
    });

    loadHomeRow(topRated, 'rating');
    loadHomeRow(mostReviewed, 'reviews');
  }

  function loadHomeRow(container, sort) {
    SS.apiGet('/api/products?sort=' + sort + '&page_size=6').then(function (data) {
      container.innerHTML = '';
      container.appendChild(productGrid(data.items || []));
    }).catch(function () {
      container.innerHTML = '';
      container.appendChild(statusNote('Could not load products.', true));
    });
  }

  /* ------------------------------------------------------ search & category */

  function listingValues(route) {
    if (route.name === 'category') {
      return {
        mode: 'category',
        category: route.segments[1] || 'other',
        q: '',
        minPrice: '',
        maxPrice: '',
        sort: route.params.sort || 'relevance',
        page: Math.max(1, parseInt(route.params.page, 10) || 1),
        title: categoryLabel(route.segments[1] || 'other')
      };
    }
    return {
      mode: 'search',
      category: route.params.category || '',
      q: route.params.q || '',
      asins: route.params.asins || '',
      minPrice: route.params.min_price || '',
      maxPrice: route.params.max_price || '',
      sort: route.params.sort || 'relevance',
      page: Math.max(1, parseInt(route.params.page, 10) || 1),
      title: route.params.q ? 'Results for “' + route.params.q + '”' : (route.params.asins ? 'Recommended products' : 'All products')
    };
  }

  function listingHash(values, page) {
    page = page || 1;
    if (values.mode === 'category') {
      var query = encodeQuery({ sort: values.sort !== 'relevance' ? values.sort : '', page: page > 1 ? page : '' });
      return '#/category/' + encodeURIComponent(values.category) + (query ? '?' + query : '');
    }
    return searchHash({
      q: values.q,
      category: values.category,
      min_price: values.minPrice,
      max_price: values.maxPrice,
      sort: values.sort !== 'relevance' ? values.sort : '',
      page: page > 1 ? page : ''
    });
  }

  function renderListing(view, values) {
    view.appendChild(pageHead(values.title, 'Loading…'));

    var categorySelect = el('select', { class: 'field-input' });
    categorySelect.appendChild(el('option', { value: '', text: 'All categories' }));
    CATEGORY_ORDER.forEach(function (name) {
      categorySelect.appendChild(el('option', { value: name, text: categoryLabel(name) }));
    });
    categorySelect.value = values.category || '';

    var qInput = el('input', { class: 'field-input', type: 'search', placeholder: 'Search products…' });
    qInput.value = values.q || '';
    var minInput = el('input', { class: 'field-input', type: 'number', min: '0', placeholder: 'Min $' });
    minInput.value = values.minPrice || '';
    var maxInput = el('input', { class: 'field-input', id: 'filterMax', type: 'number', min: '0', placeholder: 'Max $' });
    maxInput.value = values.maxPrice || '';

    var sortSelect = el('select', { class: 'field-input', id: 'sortSelect' });
    SORT_OPTIONS.forEach(function (pair) {
      sortSelect.appendChild(el('option', { value: pair[0], text: pair[1] }));
    });
    sortSelect.value = values.sort;

    function applyFilters() {
      var next = {
        mode: values.mode,
        category: categorySelect.value,
        q: qInput.value.trim(),
        minPrice: minInput.value,
        maxPrice: maxInput.value,
        sort: sortSelect.value
      };
      SS.navigate(listingHash(next, 1));
    }

    var clearHref = (values.mode === 'category' && !values.q)
      ? '#/category/' + encodeURIComponent(values.category)
      : searchHash({});

    var sidebar = el('aside', { class: 'listing-sidebar' }, [
      el('div', { class: 'filters-panel' }, [
        el('h2', { class: 'section-title', text: 'Filters' }),
        el('label', { class: 'field' }, [el('span', { class: 'field-label', text: 'Keyword' }), qInput]),
        el('label', { class: 'field' }, [el('span', { class: 'field-label', text: 'Category' }), categorySelect]),
        el('div', { class: 'field' }, [
          el('span', { class: 'field-label', text: 'Price (USD)' }),
          el('div', { class: 'field-row' }, [minInput, maxInput])
        ]),
        el('div', { class: 'field-actions' }, [
          button('Apply filters', 'btn btn-primary', applyFilters),
          link('Clear all', clearHref, 'btn btn-ghost')
        ])
      ])
    ]);
    // Enter in the keyword box applies the filters.
    qInput.addEventListener('keydown', function (event) {
      if (event.key === 'Enter') { event.preventDefault(); applyFilters(); }
    });

    sortSelect.addEventListener('change', function () {
      SS.navigate(listingHash(Object.assign({}, values, { sort: sortSelect.value }), 1));
    });

    var results = el('div', { class: 'listing-results' }, [skeletonGrid(PAGE_SIZE)]);
    view.appendChild(el('div', { class: 'listing-layout' }, [
      sidebar,
      el('div', { class: 'listing-main' }, [
        el('div', { class: 'listing-toolbar' }, [
          el('span', { class: 'listing-count', id: 'listingCount', text: 'Loading…' }),
          el('label', { class: 'sort-control' }, [el('span', { class: 'field-label', text: 'Sort by' }), sortSelect])
        ]),
        results
      ])
    ]));

    var apiParams = { page: values.page, page_size: PAGE_SIZE, sort: values.sort };
    if (values.q) { apiParams.q = values.q; }
    if (values.asins) { apiParams.asins = values.asins; }
    if (values.category) { apiParams.category = values.category; }
    if (values.minPrice !== '' && values.minPrice !== null) { apiParams.min_price = values.minPrice; }
    if (values.maxPrice !== '' && values.maxPrice !== null) { apiParams.max_price = values.maxPrice; }

    SS.apiGet('/api/products?' + encodeQuery(apiParams)).then(function (data) {
      var countNode2 = $('listingCount');
      if (countNode2) { countNode2.textContent = data.total === 1 ? '1 product' : data.total + ' products'; }
      results.innerHTML = '';
      if (!data.items || !data.items.length) {
        results.appendChild(statusNote('No products match your filters. Try widening them.'));
        return;
      }
      results.appendChild(productGrid(data.items));
      var nav = pagination(data.page || values.page, data.pages || 1, function (p) {
        return listingHash(values, p);
      });
      if (nav) { results.appendChild(nav); }
    }).catch(function () {
      results.innerHTML = '';
      results.appendChild(statusNote('Sorry, I could not load products. Please try again.', true));
      var countNode3 = $('listingCount');
      if (countNode3) { countNode3.textContent = ''; }
    });
  }

  /* -------------------------------------------------------------- product */

  function renderProduct(view, route) {
    var asin = route.segments[1];
    if (!asin) { renderNotFound(view); return; }
    view.appendChild(skeletonGrid(1));

    SS.apiGet('/api/products/' + encodeURIComponent(asin)).then(function (product) {
      view.innerHTML = '';
      view.appendChild(el('nav', { class: 'breadcrumb' }, [
        link('Home', '#/', 'crumb'),
        el('span', { class: 'crumb-sep', text: '›' }),
        link('Products', searchHash({}), 'crumb'),
        el('span', { class: 'crumb-sep', text: '›' }),
        el('span', { class: 'crumb-current', text: product.title || asin })
      ]));

      var info = el('div', { class: 'detail-info' }, [
        el('div', { class: 'detail-store', text: product.store || '' }),
        el('h1', { class: 'detail-title', text: product.title || '' }),
        starsRow(product.average_rating, product.rating_number),
        priceNode(product.price, 'detail-price'),
        el('div', { class: 'detail-actions' }, [
          button('Add to cart', 'btn btn-primary', function () { SS.cart.add(product, 1); toast('Added to cart'); }),
          button(SS.compare.has(asin) ? '✓ In compare' : 'Add to compare', 'btn btn-ghost', function () {
            if (!SS.compare.toggle(asin)) { toast('You can compare up to 3 products.'); return; }
            render();
          }),
          button('Ask ShopBot', 'btn btn-ghost', function () {
            window.ShopBot.open();
            window.ShopBot.send('Tell me about ' + product.title);
          })
        ])
      ]);

      if (product.features && product.features.length) {
        info.appendChild(el('section', { class: 'detail-section' }, [
          el('h2', { class: 'section-title', text: 'Key features' }),
          el('ul', { class: 'detail-features' }, product.features.map(function (f) { return el('li', { text: f }); }))
        ]));
      }
      if (product.description) {
        info.appendChild(el('section', { class: 'detail-section' }, [
          el('h2', { class: 'section-title', text: 'Description' }),
          el('p', { class: 'detail-description', text: product.description })
        ]));
      }

      view.appendChild(el('div', { class: 'detail-layout' }, [mediaNode(product, 'detail-media'), info]));

      var reviews = product.reviews || [];
      var reviewSection = el('section', { class: 'detail-section' }, [
        el('h2', { class: 'section-title', text: 'Top reviews' })
      ]);
      if (reviews.length) {
        reviews.forEach(function (r) {
          reviewSection.appendChild(el('div', { class: 'review-item' }, [
            el('div', { class: 'review-title', text: r.title || '' }),
            starsRow(r.rating),
            el('p', { class: 'review-text', text: r.text || '' }),
            el('div', { class: 'review-meta', text: (r.helpful_vote || 0) + ' found this helpful' })
          ]));
        });
      } else {
        reviewSection.appendChild(el('p', { class: 'detail-description', text: 'No reviews are available for this product.' }));
      }
      view.appendChild(reviewSection);
    }).catch(function () {
      view.innerHTML = '';
      view.appendChild(pageHead('Product not found'));
      view.appendChild(statusNote('We could not find that product.', true));
      view.appendChild(link('Back to all products', searchHash({}), 'btn btn-primary'));
    });
  }

  /* -------------------------------------------------------------- compare */

  function renderCompare(view, route) {
    var asins = (route.params.asins || '').split(',')
      .map(function (a) { return a.trim(); }).filter(Boolean).slice(0, 3);

    view.appendChild(pageHead('Compare products', 'Up to three products side by side.'));

    if (asins.length < 2) {
      view.appendChild(statusNote('Add at least two products to compare. Use “+ Compare” on any product.'));
      view.appendChild(link('Browse products', searchHash({}), 'btn btn-primary'));
      return;
    }

    var columns = el('div', { class: 'compare-columns', id: 'compareProducts' }, [skeletonGrid(1)]);
    var tableWrap = el('div', { class: 'compare-table-wrap', id: 'compareTable' }, [skeletonGrid(1)]);
    view.appendChild(columns);
    view.appendChild(tableWrap);

    var productRequests = asins.map(function (asin) {
      return SS.apiGet('/api/products/' + encodeURIComponent(asin)).catch(function () { return null; });
    });

    Promise.all(productRequests).then(function (products) {
      var container = $('compareProducts');
      if (!container) { return; }
      container.innerHTML = '';
      products.forEach(function (product, index) {
        var asin = asins[index];
        if (!product) {
          container.appendChild(el('div', { class: 'compare-col' }, [
            el('p', { class: 'page-note is-error', text: asin + ' is unavailable.' })
          ]));
          return;
        }
        container.appendChild(el('div', { class: 'compare-col' }, [
          mediaNode(product, 'compare-media'),
          el('a', { class: 'compare-name', href: '#/product/' + encodeURIComponent(asin), text: product.title || asin }),
          priceNode(product.price, 'compare-price')
        ]));
      });
    });

    SS.apiPost('/api/compare', { asins: asins }).then(function (data) {
      var container = $('compareTable');
      if (!container) { return; }
      container.innerHTML = '';
      container.appendChild(comparisonNode(data));
    }).catch(function () {
      var container = $('compareTable');
      if (container) {
        container.innerHTML = '';
        container.appendChild(statusNote('Sorry, I could not compare those products.', true));
      }
    });
  }

  function comparisonNode(comparison) {
    var table = comparison.table || { headers: [], rows: [] };
    var wrap = el('div', { class: 'compare-wrap' });
    var headRow = el('tr');
    (table.headers || []).forEach(function (header) { headRow.appendChild(el('th', { text: header })); });
    var bodyRows = (table.rows || []).map(function (row) {
      var tr = el('tr');
      (row || []).forEach(function (cell) { tr.appendChild(el('td', { text: cell })); });
      return tr;
    });
    wrap.appendChild(el('table', { class: 'compare-table' }, [
      el('thead', {}, [headRow]),
      el('tbody', {}, bodyRows)
    ]));
    if (comparison.summary) { wrap.appendChild(el('div', { class: 'compare-summary', text: comparison.summary })); }
    return wrap;
  }

  /* ----------------------------------------------------------------- cart */

  function renderCart(view) {
    view.appendChild(pageHead('Your cart'));
    var items = SS.cart.all();

    if (!items.length) {
      view.appendChild(statusNote('Your cart is empty.'));
      view.appendChild(link('Browse products', searchHash({}), 'btn btn-primary'));
      return;
    }

    var list = el('div', { class: 'cart-list' });
    items.forEach(function (item) {
      var qtyInput = el('input', { class: 'qty-input', type: 'number', min: '1', value: String(item.qty || 1) });
      qtyInput.addEventListener('change', function () {
        SS.cart.setQty(item.parent_asin, Math.max(1, parseInt(qtyInput.value, 10) || 1));
        render();
      });
      list.appendChild(el('div', { class: 'cart-row' }, [
        mediaNode(item, 'cart-media'),
        el('div', { class: 'cart-info' }, [
          el('a', { class: 'cart-name', href: '#/product/' + encodeURIComponent(item.parent_asin), text: item.title || item.parent_asin }),
          priceNode(item.price, 'cart-price')
        ]),
        el('label', { class: 'cart-qty' }, [el('span', { class: 'field-label', text: 'Qty' }), qtyInput]),
        button('Remove', 'btn btn-ghost', function () { SS.cart.remove(item.parent_asin); render(); })
      ]));
    });

    view.appendChild(list);

    var total = SS.cart.total();
    var totalNode = hasPrice(total)
      ? el('div', { class: 'cart-total-line' }, [
          el('span', { class: 'price-usd', text: formatUsd(total) }),
          el('span', { class: 'price-aed', text: formatAed(total) })
        ])
      : el('span', { class: 'price-unavailable', text: 'Total unavailable' });

    view.appendChild(el('div', { class: 'cart-summary' }, [
      el('div', { class: 'cart-total' }, [el('span', { class: 'field-label', text: 'Estimated total' }), totalNode]),
      statusNote('Some items may not have a listed price, so the total can be incomplete.'),
      el('div', { class: 'card-actions' }, [
        button('Clear cart', 'btn btn-ghost', function () { SS.cart.clear(); render(); }),
        link('Continue shopping', searchHash({}), 'btn btn-primary')
      ])
    ]));
  }

  /* --------------------------------------------------------------- orders */

  function renderOrders(view, route) {
    view.appendChild(pageHead('Order tracking', 'Enter an order number such as ORD-10025.'));

    var input = el('input', { class: 'field-input', id: 'orderInput', type: 'text', placeholder: 'ORD-10025' });
    input.value = route.params.order_id || '';
    var form = el('form', { class: 'order-form' }, [
      input,
      el('button', { class: 'btn btn-primary', type: 'submit', text: 'Track order' })
    ]);
    form.addEventListener('submit', function (event) {
      event.preventDefault();
      var value = input.value.trim().toUpperCase();
      if (value) { SS.navigate('#/orders?order_id=' + encodeURIComponent(value)); }
    });
    view.appendChild(form);

    var result = el('div', { class: 'order-result', id: 'orderResult' });
    view.appendChild(result);

    var orderId = route.params.order_id;
    if (!orderId) { return; }

    result.appendChild(statusNote('Looking up ' + orderId + '…'));
    SS.apiGet('/api/orders/' + encodeURIComponent(orderId)).then(function (order) {
      result.innerHTML = '';
      result.appendChild(el('div', { class: 'order-card' }, [
        el('div', { class: 'order-head' }, [
          el('span', { class: 'order-id', text: order.order_id || orderId }),
          order.status ? el('span', { class: 'order-status', text: order.status }) : null
        ]),
        el('dl', { class: 'order-fields' }, [
          orderField('Product', order.product_title),
          orderField('Quantity', order.quantity),
          orderField('Ordered', order.order_date),
          orderField('Estimated delivery', order.estimated_delivery),
          orderField('Delivered', order.delivered_date),
          orderField('Tracking number', order.tracking_number)
        ])
      ]));
    }).catch(function (error) {
      result.innerHTML = '';
      var message = (error && error.status === 404)
        ? 'We could not find order ' + orderId + '.'
        : 'Sorry, we could not look up that order right now.';
      result.appendChild(statusNote(message, true));
    });
  }

  function orderField(label, value) {
    return el('div', { class: 'order-field' }, [
      el('dt', { text: label }),
      el('dd', { text: (value === null || value === undefined || value === '') ? '—' : String(value) })
    ]);
  }

  /* ----------------------------------------------------------------- help */

  function renderHelp(view, route) {
    var topic = (route.segments[1] || 'returns').toLowerCase();
    view.appendChild(pageHead('Help & policies', 'Answers taken directly from our store policies.'));

    var tabs = el('nav', { class: 'help-tabs' });
    HELP_TOPICS.forEach(function (name) {
      tabs.appendChild(link(HELP_LABELS[name], '#/help/' + name, 'help-tab' + (name === topic ? ' is-current' : '')));
    });
    view.appendChild(tabs);

    var content = el('div', { class: 'help-content', id: 'helpContent' }, [statusNote('Loading…')]);
    view.appendChild(content);

    SS.apiGet('/api/help/' + encodeURIComponent(topic)).then(function (data) {
      content.innerHTML = SS.renderMarkdown(data.markdown || data.text || '');
    }).catch(function () {
      content.innerHTML = '';
      content.appendChild(statusNote('This help topic is not available right now.', true));
    });
  }

  /* ------------------------------------------------------------ not found */

  function renderNotFound(view) {
    view.appendChild(pageHead('Page not found'));
    view.appendChild(statusNote('That page does not exist.', true));
    view.appendChild(link('Go to the home page', '#/', 'btn btn-primary'));
  }

  /* ---------------------------------------------------------------- router */

  var RENDERERS = {
    home: renderHome,
    search: renderSearch,
    category: renderCategory,
    product: renderProduct,
    compare: renderCompare,
    cart: renderCart,
    orders: renderOrders,
    help: renderHelp,
    notfound: renderNotFound
  };

  function renderSearch(view, route) { renderListing(view, listingValues(route)); }
  function renderCategory(view, route) { renderListing(view, listingValues(route)); }

  function render() {
    var route = parseRoute();
    var view = $('view');
    if (!view) { return; }
    view.innerHTML = '';
    var renderer = RENDERERS[route.name] || renderNotFound;
    renderer(view, route);
    updateActiveTab(route);
    window.scrollTo({ top: 0, behavior: 'auto' });
  }

  /* ----------------------------------------------------------- header/nav */

  function buildCategoryTabs() {
    var nav = $('categoryTabs');
    if (!nav) { return; }
    nav.innerHTML = '';
    CATEGORY_ORDER.forEach(function (name) {
      nav.appendChild(el('a', {
        class: 'category-tab',
        href: '#/category/' + encodeURIComponent(name),
        text: categoryLabel(name)
      }));
    });
  }

  function updateActiveTab(route) {
    var current = route.name === 'category' ? route.segments[1] : null;
    document.querySelectorAll('#categoryTabs .category-tab').forEach(function (tab) {
      var href = tab.getAttribute('href') || '';
      var name = decodeURIComponent(href.replace('#/category/', ''));
      tab.classList.toggle('is-current', name === current);
    });
  }

  function updateCartCount() {
    var node = $('cartCount');
    if (node) { node.textContent = String(SS.cart.count()); }
  }

  function updateLangButton(lang) {
    var text = $('langSwitchText');
    var flag = $('langSwitchFlag');
    var isAr = lang === 'ar';
    if (text) { text.textContent = isAr ? 'AR' : 'EN'; }
    if (flag) { flag.textContent = isAr ? '🇦🇪' : '🇬🇧'; }
  }

  function updateCompareBar() {
    var bar = $('compareBar');
    if (!bar) { return; }
    var asins = SS.compare.all();
    bar.hidden = asins.length === 0;
    var text = $('compareBarText');
    if (text) { text.textContent = asins.length + (asins.length === 1 ? ' product selected' : ' products selected'); }
    var go = $('compareBarGo');
    if (go) { go.setAttribute('href', compareHash(asins)); }
    var clear = $('compareBarClear');
    if (clear) { clear.disabled = asins.length === 0; }
  }

  /* ---------------------------------------------------------------- wiring */

  function wire() {
    var form = $('headerSearchForm');
    if (form) {
      form.addEventListener('submit', function (event) {
        event.preventDefault();
        SS.navigate(searchHash({ q: $('headerSearchInput').value.trim() }));
      });
    }

    var chatBtn = $('headerChatBtn');
    if (chatBtn) { chatBtn.addEventListener('click', function () { window.ShopBot.open(); }); }

    // The header language toggle drives the assistant's language (the storefront
    // itself is English-only). Kept in sync with the chat's own toggle.
    var langBtn = $('langSwitcher');
    if (langBtn && window.ShopBot) {
      updateLangButton(window.ShopBot.getLang());
      langBtn.addEventListener('click', function () {
        var next = window.ShopBot.getLang() === 'ar' ? 'en' : 'ar';
        window.ShopBot.setLang(next);
        updateLangButton(next);
      });
      window.addEventListener('shopbot:lang', function (event) { updateLangButton(event.detail); });
    }

    var clear = $('compareBarClear');
    if (clear) { clear.addEventListener('click', function () { SS.compare.clear(); render(); }); }

    SS.on('cart', updateCartCount);
    SS.on('compare', function () { updateCompareBar(); render(); });
    SS.on('route', render);

    document.addEventListener('keydown', function (event) {
      if (event.key !== '/') { return; }
      var active = document.activeElement;
      if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) { return; }
      event.preventDefault();
      var input = $('headerSearchInput');
      if (input) { input.focus(); }
    });
  }

  /* ------------------------------------------------------------------ init */

  function init() {
    buildCategoryTabs();
    updateCartCount();
    updateCompareBar();

    // Hand the assistant the store's navigation + page context.
    if (window.ShopBot && typeof window.ShopBot.init === 'function') {
      window.ShopBot.init({
        apiBase: SS.API_BASE,
        host: {
          navigate: function (url) { SS.navigate(url); },
          getContext: assistantContext
        }
      });
    }

    wire();

    window.addEventListener('hashchange', function () {
      updateCompareBar();
      render();
    });

    if (!window.location.hash) { window.location.hash = '#/'; }
    else { render(); }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
