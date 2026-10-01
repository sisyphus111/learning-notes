const themeButton = document.querySelector('.theme-toggle');
function updateThemeLabel() {
    themeButton.textContent = document.documentElement.dataset.theme === 'dark' ? '浅色' : '深色';
}
updateThemeLabel();
themeButton.addEventListener('click', () => {
    const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('notes-theme', theme); } catch (error) { /* Storage may be disabled. */ }
    updateThemeLabel();
});
const menuButton = document.querySelector('.menu-toggle');
menuButton.addEventListener('click', () => {
    const open = document.body.classList.toggle('nav-open');
    menuButton.setAttribute('aria-expanded', String(open));
});
document.addEventListener('click', (event) => {
    if (document.body.classList.contains('nav-open') && !event.target.closest('.sidebar, .menu-toggle')) {
        document.body.classList.remove('nav-open');
        menuButton.setAttribute('aria-expanded', 'false');
    }
});
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
    dialog.showModal();
    input.focus();
    status.textContent = searchIndex ? '输入题号、题目或 STL 用法' : '正在加载搜索索引…';
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
    if (!query) { status.textContent = '输入题号、题目或 STL 用法'; return; }
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
const tocLinks = [...document.querySelectorAll('.toc a')];
const byId = new Map(tocLinks.map((link) => [decodeURIComponent(link.hash.slice(1)), link]));
if (byId.size) {
    const observer = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            for (const link of tocLinks) link.classList.remove('active');
            byId.get(entry.target.id)?.classList.add('active');
        }
    }, { rootMargin: '-80px 0px -65% 0px' });
    for (const id of byId.keys()) { const heading = document.getElementById(id); if (heading) observer.observe(heading); }
}
