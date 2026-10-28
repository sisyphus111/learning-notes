const themeButton = document.querySelector('.theme-toggle');
function updateThemeLabel() {
    const dark = document.documentElement.dataset.theme === 'dark';
    const label = dark ? '切换到浅色模式' : '切换到深色模式';
    themeButton.setAttribute('aria-pressed', String(dark));
    themeButton.setAttribute('aria-label', label);
    themeButton.title = label;
}
updateThemeLabel();
themeButton.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    document.documentElement.dataset.themeSource = 'manual';
    try { localStorage.setItem('notes-theme', theme); } catch (error) { /* Storage may be disabled. */ }
    updateThemeLabel();
});
// Follow the OS until the reader explicitly chooses a theme.
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (event) => {
    if (document.documentElement.dataset.themeSource !== 'system') return;
    document.documentElement.dataset.theme = event.matches ? 'dark' : 'light';
    updateThemeLabel();
});
for (const example of document.querySelectorAll('.code-example')) {
    const tabs = [...example.querySelectorAll('[role="tab"]')];
    function selectTab(selected) {
        for (const tab of tabs) {
            const active = tab === selected;
            tab.setAttribute('aria-selected', String(active));
            tab.tabIndex = active ? 0 : -1;
            document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
        }
    }
    for (const tab of tabs) {
        tab.addEventListener('click', () => selectTab(tab));
        tab.addEventListener('keydown', (event) => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
            event.preventDefault();
            let index = tabs.indexOf(tab);
            if (event.key === 'Home') index = 0;
            else if (event.key === 'End') index = tabs.length - 1;
            else index = (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
            selectTab(tabs[index]);
            tabs[index].focus();
        });
    }
}
const menuButton = document.querySelector('.menu-toggle');
const sidebar = document.querySelector('.sidebar');
const menuClose = document.createElement('button');
menuClose.className = 'menu-close';
menuClose.type = 'button';
menuClose.textContent = '关闭导航 ×';
sidebar.prepend(menuClose);
function closeMenu() {
    document.body.classList.remove('nav-open');
    menuButton.setAttribute('aria-expanded', 'false');
}
menuButton.addEventListener('click', () => {
    const open = document.body.classList.toggle('nav-open');
    menuButton.setAttribute('aria-expanded', String(open));
    if (open) menuClose.focus();
});
menuClose.addEventListener('click', () => {
    closeMenu();
    menuButton.focus();
});
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && document.body.classList.contains('nav-open')) {
        closeMenu();
        menuButton.focus();
    }
});
window.matchMedia('(max-width: 1024px)').addEventListener('change', (event) => {
    if (!event.matches) closeMenu();
});
document.addEventListener('click', (event) => {
    if (document.body.classList.contains('nav-open') && !event.target.closest('.sidebar, .menu-toggle')) {
        closeMenu();
    }
});
// Reuse the existing chapter links when the desktop side TOC is hidden.
const chapterNav = document.querySelector('.toc nav');
if (chapterNav) {
    const contents = document.createElement('details');
    contents.className = 'mobile-toc';
    contents.innerHTML = '<summary>本页目录</summary>';
    contents.append(chapterNav.cloneNode(true));
    document.querySelector('article').before(contents);
    contents.addEventListener('click', (event) => {
        const link = event.target.closest('a');
        if (!link) return;
        // Close before the anchor scroll so the destination keeps its position.
        contents.open = false;
        const heading = document.getElementById(decodeURIComponent(link.hash.slice(1)));
        if (heading) {
            heading.tabIndex = -1;
            heading.focus({preventScroll: true});
        }
    });
}
// Show a scroll hint and keyboard stop only when the table is actually wider.
const tableObserver = new ResizeObserver((entries) => {
    for (const { target } of entries) {
        const viewport = target.matches('.table-scroll') ? target : target.parentElement;
        const overflowing = viewport.scrollWidth > viewport.clientWidth + 1;
        viewport.classList.toggle('is-overflowing', overflowing);
        viewport.tabIndex = overflowing ? 0 : -1;
    }
});
for (const viewport of document.querySelectorAll('.table-scroll')) {
    tableObserver.observe(viewport);
    tableObserver.observe(viewport.querySelector('table'));
}
// Preserve readable diagram labels on narrow screens with a local scroll area.
for (const image of document.querySelectorAll('article img[src$=".svg"]')) {
    const figure = document.createElement('figure');
    figure.className = 'reading-diagram';
    const viewport = document.createElement('div');
    viewport.className = 'diagram-scroll';
    viewport.tabIndex = 0;
    viewport.setAttribute('role', 'region');
    viewport.setAttribute('aria-label', `${image.alt}；可横向滚动`);
    const caption = document.createElement('figcaption');
    const hint = document.createElement('span');
    hint.textContent = '左右滑动查看图示';
    const original = document.createElement('a');
    original.href = image.getAttribute('src');
    original.textContent = '查看原图 ↗';
    caption.append(hint, original);
    const parent = image.parentElement;
    if (parent.tagName === 'P' && parent.childNodes.length === 1) parent.replaceWith(figure);
    else image.replaceWith(figure);
    viewport.append(image);
    figure.append(viewport, caption);
}
for (const block of document.querySelectorAll('.codehilite')) {
    const button = document.createElement('button');
    button.className = 'copy-button';
    button.textContent = '复制';
    button.setAttribute('aria-label', '复制此代码块');
    button.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(block.querySelector('pre').textContent);
            button.textContent = '已复制';
        } catch (error) {
            button.textContent = '请选中复制';
        }
        setTimeout(() => { button.textContent = '复制'; }, 1800);
    });
    block.append(button);
}
const dialog = document.querySelector('#search-dialog');
const input = document.querySelector('#search-input');
const status = document.querySelector('#search-status');
const results = document.querySelector('#search-results');
let searchIndex = null;
let indexPromise = null;
async function loadIndex() {
    if (!indexPromise) {
        indexPromise = fetch('assets/search-index.json').then((response) => {
            if (!response.ok) throw new Error('Search index unavailable');
            return response.json();
        }).catch((error) => { indexPromise = null; throw error; });
    }
    searchIndex = await indexPromise;
}
async function openSearch() {
    closeMenu();
    dialog.showModal();
    input.focus();
    status.textContent = searchIndex ? '输入题号、题目或容器 / API 名称' : '正在加载搜索索引…';
    try { await loadIndex(); search(); }
    catch (error) { status.textContent = '搜索索引加载失败，请刷新后重试。'; }
}
document.querySelector('.search-toggle').addEventListener('click', openSearch);
document.addEventListener('keydown', (event) => {
    const typing = event.target.closest('input, textarea, [contenteditable]');
    if (event.key === '/' && !typing && !dialog.open) { event.preventDefault(); openSearch(); }
});
function search() {
    results.replaceChildren();
    const query = input.value.trim().toLowerCase();
    if (!query) { status.textContent = '输入题号、题目或容器 / API 名称'; return; }
    if (!searchIndex) return;
    const terms = query.split(/\s+/);
    const matches = searchIndex.map((item) => {
        const title = item.title.toLowerCase();
        const body = `${item.chapter} ${item.text}`.toLowerCase();
        if (!terms.every((term) => title.includes(term) || body.includes(term))) return null;
        const score = title === query ? 100 : title.startsWith(query + '.') ? 90 : title.includes(query) ? 80 : 10;
        return { item, score };
    }).filter(Boolean).sort((a, b) => b.score - a.score);
    status.textContent = matches.length ? `找到 ${matches.length} 个章节，显示前 ${Math.min(40, matches.length)} 个` : '没有匹配结果，试试题号或更短的关键词。';
    for (const { item } of matches.slice(0, 40)) {
        const link = document.createElement('a');
        link.className = 'search-result';
        link.href = item.url;
        const chapter = document.createElement('small');
        chapter.textContent = item.chapter;
        const title = document.createElement('span');
        title.textContent = item.title;
        link.append(chapter, title);
        link.addEventListener('click', () => { dialog.close(); });
        results.append(link);
    }
}
let debounce;
input.addEventListener('input', () => { clearTimeout(debounce); debounce = setTimeout(search, 100); });
dialog.addEventListener('click', (event) => {
    if (event.target === dialog) {
        const bounds = dialog.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
    }
});
const tocItems = [...document.querySelectorAll('.toc a')].map((link) => ({
    link,
    heading: document.getElementById(decodeURIComponent(link.hash.slice(1))),
})).filter((item) => item.heading);
if (tocItems.length) {
    const topbar = document.querySelector('.topbar');
    let currentItem;
    let pendingFrame = 0;
    let revealCurrent = true;
    function updateToc() {
        // The current section starts above the reading line and ends at the next heading.
        // Read live positions so images, formulas and code tabs cannot stale an offset cache.
        const readingLine = Math.max(80, topbar.getBoundingClientRect().bottom + 16);
        let low = 0;
        let high = tocItems.length;
        while (low < high) {
            const middle = (low + high) >>> 1;
            if (tocItems[middle].heading.getBoundingClientRect().top <= readingLine + 1) low = middle + 1;
            else high = middle;
        }
        // A short final section may never reach the reading line before the page ends.
        const atBottom = window.scrollY > 0 &&
            Math.ceil(window.scrollY + window.innerHeight) >= document.documentElement.scrollHeight - 1;
        const nextItem = tocItems[atBottom ? tocItems.length - 1 : Math.max(0, low - 1)];
        if (nextItem !== currentItem) {
            if (currentItem) {
                currentItem.link.classList.remove('active');
                currentItem.link.removeAttribute('aria-current');
            }
            nextItem.link.classList.add('active');
            nextItem.link.setAttribute('aria-current', 'location');
            currentItem = nextItem;
            revealCurrent = true;
        }
        // Reveal on section changes, leaving manual directory scrolling alone otherwise.
        // Scroll only the directory; scrollIntoView would also move the article.
        if (revealCurrent && chapterNav.clientHeight) {
            const bounds = chapterNav.getBoundingClientRect();
            const linkBounds = currentItem.link.getBoundingClientRect();
            if (linkBounds.top < bounds.top) chapterNav.scrollTop += linkBounds.top - bounds.top - 8;
            else if (linkBounds.bottom > bounds.bottom) chapterNav.scrollTop += linkBounds.bottom - bounds.bottom + 8;
            revealCurrent = false;
        }
    }
    function scheduleTocUpdate() {
        if (pendingFrame) return;
        pendingFrame = requestAnimationFrame(() => {
            pendingFrame = 0;
            updateToc();
        });
    }
    window.addEventListener('scroll', scheduleTocUpdate, { passive: true });
    window.addEventListener('resize', () => {
        revealCurrent = true;
        scheduleTocUpdate();
    });
    window.addEventListener('hashchange', scheduleTocUpdate);
    window.addEventListener('pageshow', scheduleTocUpdate);
    new ResizeObserver(scheduleTocUpdate).observe(document.querySelector('article'));
    scheduleTocUpdate();
}
