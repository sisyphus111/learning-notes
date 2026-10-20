/* Deterministic teaching models, not GPU timing or hardware simulations.
 * Every panel is appended to an existing section; Markdown and heading IDs stay untouched.
 */
(() => {
  'use strict';
  const article = document.querySelector('article');
  if (!article) return;
  const panels = [];
  const sum = (xs) => xs.reduce((a, b) => a + b, 0);
  const seq = (n) => Array.from({length: n}, (_, i) => i);
  const aValue = (m, k) => (m + k) % 4 + 1;
  const bValue = (k, n) => (k + 2 * n) % 3 + 1;
  const dot = (m, n, end, start = 0) => sum(seq(end - start).map(i => aValue(m, start + i) * bValue(start + i, n)));

  function matrix(label, rows, cols, value, state = () => '') {
    return `<div class="gd-matrix"><span class="gd-label">${label}</span>
      <div class="gd-grid" style="--cols:${cols}" role="group" aria-label="${label}">
      ${seq(rows).map(r => seq(cols).map(c =>
        `<span class="gd-cell ${state(r, c)}" aria-label="第 ${r} 行，第 ${c} 列：${value(r, c)}">${value(r, c)}</span>`
      ).join('')).join('')}</div></div>`;
  }
  const option = (name, label, values) => ({name, label, values});
  function mount(config) {
    const heading = [...article.querySelectorAll('h2,h3')].find(h => h.textContent.startsWith(config.section + ' '));
    if (!heading) throw new Error(`Missing GEMM section ${config.section}`);
    let next = heading.nextElementSibling;
    while (next && !/^H[23]$/.test(next.tagName)) next = next.nextElementSibling;
    const el = document.createElement('details');
    el.className = 'gemm-demo';
    el.id = `gemm-${config.id}`;
    el.open = true;
    el.innerHTML = `<summary>动画 · ${config.title}</summary><div class="gd-body">
      <p class="gd-intro">${config.intro}</p>
      <div class="gd-options">${(config.options || []).map(o => `<label>${o.label}
        <select data-option="${o.name}" aria-label="${config.title}：${o.label}">
        ${o.values.map(([value, label]) => `<option value="${value}">${label}</option>`).join('')}</select></label>`).join('')}</div>
      <div class="gd-controls">
        <button type="button" class="gd-play" aria-pressed="false">播放</button>
        <button type="button" data-action="prev" aria-label="${config.title}：上一步">上一步</button>
        <button type="button" data-action="next" aria-label="${config.title}：下一步">下一步</button>
        <button type="button" data-action="reset">重置</button>
        <label>速度 <select class="gd-speed" aria-label="${config.title}：播放速度"><option value="1300">慢速</option><option value="700">正常</option></select></label>
      </div>
      <div class="gd-progress"><input type="range" min="0" value="0" step="1" aria-label="${config.title}：动画进度"><span class="gd-counter"></span></div>
      <div class="gd-scene"></div><p class="gd-status" role="status" aria-live="polite"></p>
      ${config.note ? `<p class="gd-note">${config.note}</p>` : ''}
      </div>`;
    article.insertBefore(el, next);
    const scene = el.querySelector('.gd-scene');
    const status = el.querySelector('.gd-status');
    const slider = el.querySelector('input');
    const play = el.querySelector('.gd-play');
    const prev = el.querySelector('[data-action=prev]');
    const forward = el.querySelector('[data-action=next]');
    const state = {step: 0, options: Object.fromEntries((config.options || []).map(o => [o.name, o.values[0][0]]))};
    let timer = null;
    const max = () => typeof config.steps === 'function' ? config.steps(state) : config.steps;
    function pause() {
      clearTimeout(timer); timer = null; play.textContent = '播放'; play.setAttribute('aria-pressed', 'false');
    }
    function render() {
      slider.max = max(); slider.value = state.step;
      el.querySelector('.gd-counter').textContent = `${state.step} / ${max()}`;
      prev.disabled = state.step === 0; forward.disabled = state.step === max();
      status.textContent = config.draw(scene, state);
    }
    function tick() {
      if (state.step >= max()) { pause(); return; }
      timer = setTimeout(() => {
        state.step++; render();
        if (state.step === max()) pause(); else tick();
      }, Number(el.querySelector('.gd-speed').value));
    }
    play.addEventListener('click', () => {
      if (timer !== null) { pause(); return; }
      panels.forEach(p => p.pause());
      if (state.step === max()) { state.step = 0; render(); }
      play.textContent = '暂停'; play.setAttribute('aria-pressed', 'true'); tick();
    });
    for (const button of [prev, forward, el.querySelector('[data-action=reset]')]) {
      button.addEventListener('click', () => {
        pause();
        state.step = button === prev ? Math.max(0, state.step - 1) : button === forward ? Math.min(max(), state.step + 1) : 0;
        render();
      });
    }
    slider.addEventListener('input', () => { pause(); state.step = Number(slider.value); render(); });
    el.querySelectorAll('[data-option]').forEach(select => select.addEventListener('change', () => {
      pause(); state.options[select.dataset.option] = select.value; state.step = 0; render();
    }));
    el.addEventListener('toggle', () => { if (!el.open) pause(); });
    // Leave the current frame in place when a reader scrolls away or changes tabs.
    const observer = new IntersectionObserver(entries => {
      if (!entries[0].isIntersecting) pause();
    });
    observer.observe(el);
    panels.push({pause});
    render();
  }
  document.addEventListener('visibilitychange', () => { if (document.hidden) panels.forEach(p => p.pause()); });

  mount({
    id: 'dot', section: '1.', title: '一行 × 一列，怎样得到一个输出',
    intro: '选择 D 的一个位置，沿 K 看两边的数逐对相乘。蓝色是 A 的行，橙色是 B 的列，绿色是当前输出。',
    options: [option('m', '输出行 m', [[0, '0'], [1, '1'], [2, '2']]), option('n', '输出列 n', [[0, '0'], [1, '1'], [2, '2']])],
    steps: 6,
    draw(scene, {step, options}) {
      const m = +options.m, n = +options.n, k = step - 1;
      scene.innerHTML = `<div class="gd-matrices">
        ${matrix('A [3,6] · 行 m', 3, 6, aValue, (r,c) => r === m ? 'is-a' + (c === k ? ' is-now' : '') : '')}
        ${matrix('B [6,3] · 列 n', 6, 3, bValue, (r,c) => c === n ? 'is-b' + (r === k ? ' is-now' : '') : '')}
        ${matrix('D [3,3] · 输出', 3, 3, (r,c) => r === m && c === n ? dot(m,n,step) : '·', (r,c) => r === m && c === n ? 'is-c' : '')}
        </div><div class="gd-formula">${step ? `${dot(m,n,k)} + A[${m},${k}] × B[${k},${n}] = ${dot(m,n,k)} + ${aValue(m,k)} × ${bValue(k,n)} = ` : `D[${m},${n}] 的累加器 = `}<strong>${dot(m,n,step)}</strong></div>`;
      return step === 6 ? `六项已累加完，D[${m},${n}] = ${dot(m,n,6)}。更换 m 或 n，就在计算另一个独立输出。` : step ? `正在加入 k=${k} 的乘积。输出位置一直是 [${m},${n}]，变化的是 K 上取数的位置。` : '先固定一个输出位置。点击“下一步”，加入第一项乘积。';
    }
  });

  mount({
    id: 'k-segments', section: '2.', title: '把 91 拆成三个部分和',
    intro: '沿用正文的六个数：每段两个元素，段内先求和，再加入同一个累加器。',
    steps: 3,
    draw(scene, {step}) {
      const partial = [5,25,61];
      scene.innerHTML = `<div class="gd-parts">${partial.map((p,i) => `<div class="gd-part ${i < step ? 'is-current' : ''}">K 段 ${i}<strong>${2*i+1}² + ${2*i+2}² = ${i < step ? p : '?'}</strong></div>`).join('')}</div>
        <div class="gd-formula">同一个 acc：${step ? partial.slice(0,step).join(' + ') : '0'} = <strong>${sum(partial.slice(0,step))}</strong></div>`;
      return step === 3 ? '5 + 25 + 61 = 91。拆 K 没有增加输出个数，三个部分和都属于同一个输出。' : `已经处理 ${step * 2} / 6 个 K 元素；下一段继续更新原来的 acc。`;
    }
  });

  mount({
    id: 'tiles', section: '4.', title: '固定一块输出，让 A / B 的小块沿 K 移动',
    intro: '小例子 M=N=4、K=6，输出 tile 为 2×2。选择另一块输出，或改变每轮处理的 BK。',
    options: [option('tile','输出 tile',[[0,'左上'],[1,'右上'],[2,'左下'],[3,'右下']]), option('bk','BK',[[2,'2'],[1,'1'],[3,'3']])],
    steps: ({options}) => 6 / +options.bk,
    draw(scene, {step, options}) {
      const bk = +options.bk, m0 = Math.floor(+options.tile/2)*2, n0 = (+options.tile%2)*2;
      const start = Math.max(0,step-1)*bk, end = step*bk;
      scene.innerHTML = `<div class="gd-matrices">
        ${matrix('A [4,6] · 本轮 2×BK',4,6,aValue,(r,c) => r>=m0 && r<m0+2 && c>=start && c<start+bk ? 'is-a' : '')}
        ${matrix('B [6,4] · 本轮 BK×2',6,4,bValue,(r,c) => c>=n0 && c<n0+2 && r>=start && r<start+bk ? 'is-b' : '')}
        ${matrix('D [4,4] · 同一块 acc',4,4,(r,c) => r>=m0 && r<m0+2 && c>=n0 && c<n0+2 ? dot(r,c,end) : '·',(r,c) => r>=m0 && r<m0+2 && c>=n0 && c<n0+2 ? 'is-c' : '')}
        </div><div class="gd-formula">4 个输出 tiles → 基础分工 4 个 CTAs<br>每个 CTA 内：${6} / ${bk} = ${6/bk} 轮 K 循环；已完成 ${step} 轮</div>`;
      return step ? `本轮取 K=[${start},${end})，累加到 D 的行 [${m0},${m0+2})、列 [${n0},${n0+2})。${end === 6 ? '这一块输出已完成。' : '下一轮仍然更新这块输出。'}` : '高亮的是即将加载的第一对输入块。绿色 acc 从零开始。';
    }
  });

  mount({
    id:'reuse', section:'5.', title:'同一份输入，反复服务多个输出',
    intro:'A[3,2] × B[2,3]。逐个查看 9 个输出；每个输出需要 2 个 A 值、2 个 B 值。',
    steps:9,
    note:'读取计数是教学模型：比较独立请求与协作加载，不代表实测 DRAM 流量；cache 可能吸收重复请求。',
    draw(scene,{step}) {
      const i = Math.max(0,step-1), m = Math.floor(i/3), n = i%3;
      scene.innerHTML = `<div class="gd-matrices">${matrix('A：6 个输入元素',3,2,aValue,(r) => step && r===m?'is-a':'')}${matrix('B：6 个输入元素',2,3,bValue,(_,c) => step && c===n?'is-b':'')}${matrix('D：9 个输出',3,3,(r,c)=>r*3+c<step?dot(r,c,2):'·',(r,c)=>r===m&&c===n&&step?'is-c':'')}</div>
        <div class="gd-parts" style="margin-top:16px"><div class="gd-part">各输出独立加载<strong>${step} × 4 = ${step*4} 次元素请求</strong></div><div class="gd-part">先协作加载整个输入 tile<strong>6 + 6 = 12 个元素</strong>后续从 shared memory 复用</div></div>`;
      return step ? `D[${m},${n}] 读取高亮的 A 行和 B 列。A 的每个值用于 3 个输出，B 的每个值也用于 3 个输出。` : '协作加载准备好 12 个输入元素后，计算线程可以反复使用它们。';
    }
  });

  mount({
    id:'registers',section:'6.',title:'四个寄存器中的累加器，一起更新',
    intro:'一个线程负责 2×2 输出。每轮读 2 个 A 值和 2 个 B 值，形成一个 outer product。',
    steps:3,
    draw(scene,{step}) {
      const k=Math.max(0,step-1);
      scene.innerHTML=`<div class="gd-matrices">${matrix('A 的两行',2,3,aValue,(_,c)=>step&&c===k?'is-a':'')}${matrix('B 的两列',3,2,bValue,r=>step&&r===k?'is-b':'')}${matrix('4 个 acc',2,2,(r,c)=>dot(r,c,step),()=> 'is-c')}</div>
        <div class="gd-equations">${seq(2).map(r=>seq(2).map(c=>`<span>acc${r}${c}：${step?`${dot(r,c,k)} + ${aValue(r,k)}×${bValue(k,c)} = `:''}<b>${dot(r,c,step)}</b></span>`).join('')).join('')}</div>`;
      return step ? `k=${k}：同一个 A 值用于两个输出列，同一个 B 值用于两个输出行。${step===3?'三轮后得到完整的 2×2 结果。':''}` : '四个累加器都从 0 开始。一次“下一步”同时更新四个位置。';
    }
  });

  mount({
    id:'tmem',section:'7.',title:'输入块不断更换，累加结果留在 TMEM',
    intro:'以 SM100 的一种数据流为例：A/B 输入块放在 shared memory；每轮 MMA 更新同一块 TMEM 累加器。最后才读到寄存器做后处理。',
    steps:8,
    note:'缩小为 2×2 输出、3 段 K 以便手算；这里的尺寸不是 tcgen05 指令的合法形状说明。每个“计算完成”步骤包含所需的完成等待。',
    draw(scene,{step}) {
      const loaded=Math.min(3,Math.ceil(step/2)),done=Math.min(3,Math.floor(step/2));
      const k0=Math.max(0,loaded-1)*2;
      scene.innerHTML=`<div class="gd-matrices">
        ${matrix(`Shared A · K段 ${Math.max(0,loaded-1)}`,2,2,(r,c)=>loaded?aValue(r,k0+c):'·',()=>loaded?'is-a':'')}
        ${matrix(`Shared B · K段 ${Math.max(0,loaded-1)}`,2,2,(r,c)=>loaded?bValue(k0+r,c):'·',()=>loaded?'is-b':'')}
        ${matrix('同一块 TMEM acc',2,2,(r,c)=>dot(r,c,done*2),()=> 'is-c')}
        </div><div class="gd-flow">
          <span class="gd-node ${step>=7?'is-s':''}">普通寄存器<br>${step>=7?'已读取完整结果':'尚未读取 TMEM'}</span>
          <span class="gd-arrow">→ 后处理 →</span>
          <span class="gd-node ${step===8?'is-c':''}">Global 输出<br>${step===8?'已写回':'尚未写回'}</span>
        </div><div class="gd-formula">已完成 ${done} / 3 段 K；acc[0,0] = ${dot(0,0,done*2)}</div>`;
      if(step===0) return '先准备输入缓冲和 TMEM 累加空间。';
      if(step===7) return '全部 K 段的 MMA 已完成；线程从 TMEM 读取完整结果到普通寄存器，并等待读取完成。';
      if(step===8) return '后处理完成后写回 global memory。中间每段的累加结果没有反复写回显存。';
      return step%2?`第 ${loaded-1} 段输入已加载。旧的 acc 保留在 TMEM 中，等待这一轮 MMA 更新。`:`第 ${done-1} 段 MMA 已完成。TMEM 保存累加至今的结果，下一轮更换输入块后继续累加。`;
    }
  });

  mount({
    id:'coalescing',section:'8.1',title:'把线程访问的位置，摊到物理地址上看',
    intro:'只画一个 warp 的前 8 个 lanes；row-major FP32 矩阵每行 8 个元素。每格是一个元素，下方数字是元素偏移。',
    options:[option('mode','访问方式',[['row','相邻 lanes 读同一行'],['col','相邻 lanes 读同一列']])],
    steps:8,
    note:'逐步点亮是为了辨认地址；真实模型是这些 lanes 在同一条访存指令中发起访问。按 32-byte 对齐起点计算覆盖的 sectors。',
    draw(scene,{step,options}) {
      const column=options.mode==='col';
      const offsets=seq(step).map(i=>column?i*8:i);
      scene.innerHTML=`<div class="gd-memory">${seq(64).map(i=>`<span class="gd-cell ${offsets.includes(i)?'is-a':''}">${offsets.includes(i)?`L${offsets.indexOf(i)}`:i}</span>`).join('')}</div>
        <div class="gd-formula">字节地址偏移：${offsets.length?offsets.map(i=>i*4).join('，'):'等待访问'}<br>覆盖 32-byte sectors：<strong>${column?step:step?1:0}</strong></div>`;
      return column?'相邻 lanes 的地址相差 32 bytes：每个 lane 落在不同的 sector。':'相邻 lanes 的地址相差 4 bytes：这 8 个元素可以落在同一个 32-byte sector。';
    }
  });

  mount({
    id:'banks',section:'8.2',title:'同样读一列，padding 怎样改变 bank 映射',
    intro:'经典 FP32 模型：32 个 banks，bank = 元素偏移 % 32。这里读取第 0 列，各 lane 负责一行。',
    options:[option('mode','布局',[[32,'tile[32][32]'],[33,'tile[32][33]（padding）'],[0,'所有 lanes 读同一地址（广播）']])],
    steps:32,
    draw(scene,{step,options}) {
      const stride=+options.mode;
      scene.innerHTML=`<div class="gd-bank">${seq(32).map(i=>`<span class="${i<step?(stride===32?'is-b':'is-a'):''}">lane ${i}<small>${i<step?`→ bank ${(i*stride)%32}`:'待显示'}</small></span>`).join('')}</div>
        <div class="gd-formula">${stride?`元素偏移 = lane × ${stride}；bank = (lane × ${stride}) % 32`:'元素偏移始终是 0；所有线程读取同一个值'}</div>`;
      return stride===32?'所有线程访问不同地址，却都映射到 bank 0：形成冲突。':stride===33?'行跨度从 32 变成 33，bank 依次是 0、1、2……31；逻辑上仍在取第 0 列。':'同一地址的广播与“不同地址命中同一 bank”不同，不能把它也当成 32 路冲突。';
    }
  });

  const serialLoads=seq(4).map(i=>[i*5,2,i]);
  const serialComputes=seq(4).map(i=>[i*5+2,3,i]);
  const overlapLoads=[[0,2,0],[2,2,1],[5,2,2],[8,2,3]];
  const overlapComputes=seq(4).map(i=>[2+i*3,3,i]);
  function track(label, events, type, time) {
    return `<div class="gd-track"><span class="gd-track-label">${label}</span><div class="gd-rail">
      ${events.map(([start,duration,id])=>`<span class="gd-bar is-${type==='L'?'a':'c'} ${time>=start?'is-started':''}" style="left:${start*5}%;width:${duration*5}%" aria-label="${type}${id}，从 ${start} 到 ${start+duration}">${type}${id}</span>`).join('')}
      <span class="gd-cursor" style="left:${time*5}%"></span></div></div>`;
  }
  mount({
    id:'pipeline',section:'9.',title:'双缓冲：加载和计算在同一条时间轴上相遇',
    intro:'L 是加载，C 是计算，下标是 K tile 编号。假设加载需 2 个时间单位、计算需 3 个；拖动游标观察两个槽何时可复用。',
    steps:20,
    note:'理想化依赖示意，不是硬件性能数据。每槽必须加载完成后才能读、计算完成后才能覆盖。',
    draw(scene,{step}) {
      if (!scene.firstChild) scene.innerHTML=`<div class="gd-timeline">${track('串行 / 加载',serialLoads,'L',step)}${track('串行 / 计算',serialComputes,'C',step)}${track('双缓冲 / 加载',overlapLoads,'L',step)}${track('双缓冲 / 计算',overlapComputes,'C',step)}</div>
        <div class="gd-slots"></div><div class="gd-formula"></div>`;
      // Keep timeline nodes alive so the cursor moves between ticks rather than jumping.
      scene.querySelectorAll('.gd-track').forEach((row, i) => {
        const events=[serialLoads,serialComputes,overlapLoads,overlapComputes][i];
        row.querySelector('.gd-cursor').style.left=`${step*5}%`;
        row.querySelectorAll('.gd-bar').forEach((bar,j)=>bar.classList.toggle('is-started',step>=events[j][0]));
      });
      scene.querySelector('.gd-slots').innerHTML=seq(2).map(slot=>{
          const candidates=overlapLoads.filter(e=>e[2]%2===slot&&e[0]<=step);
          const e=candidates[candidates.length-1];
          let label='空闲',kind='';
          if(e) {
            const [start,duration,id]=e, compute=overlapComputes[id];
            if(step<start+duration) {label=`tile ${id} 加载中`;kind='is-a';}
            else if(step<compute[0]) {label=`tile ${id} 已就绪，等计算`;kind='is-s';}
            else if(step<compute[0]+compute[1]) {label=`tile ${id} 计算中，不可覆盖`;kind='is-c';}
            else label=`tile ${id} 已用完，可复用`;
          }
          return `<div class="gd-slot ${kind}"><b>buffer ${slot}</b>${label}</div>`;
        }).join('');
      scene.querySelector('.gd-formula').textContent=`时间 t=${step}　｜　串行结束 t=20；双缓冲结束 t=14`;
      return step<2?'先加载 tile 0，计算暂时等待。':step<14?'两个缓冲槽轮流使用：tile i 放在 slot i % 2；输入搬运可以和另一槽的计算重叠。':step<20?'双缓冲已完成四块；串行路径仍有工作。启动与排空也占时间。':'两条路径完成相同的四个 K tiles，只改变了搬运与计算的组织方式。';
    }
  });

  mount({
    id:'tma',section:'10.1',title:'从一个 tile 请求，到 shared memory 数据就绪',
    intro:'示意矩阵 [6,8]；把行 [2,4)、列 [4,8) 的 2×4 tile 搬进 shared memory。按步骤看请求、搬运、完成和消费。',
    steps:4,
    note:'流程示意省略具体指令和对齐设置；tensor descriptor 不会替 kernel 选择 GEMM 分块。',
    draw(scene,{step}) {
      if(!scene.firstChild) scene.innerHTML=`<div class="gd-matrices">
        ${matrix('Global memory [6,8]',6,8,(r,c)=>r*8+c,(r,c)=>r>=2&&r<4&&c>=4?'is-a':'')}
        <div class="gd-transfer" aria-hidden="true"><span class="gd-packet"></span></div>
        <div data-target></div></div><div class="gd-formula" data-barrier></div>`;
      scene.querySelector('[data-target]').innerHTML=matrix('Shared memory [2,4]',2,4,(r,c)=>step>=3?(r+2)*8+c+4:'·',()=>step>=3?'is-c':'');
      scene.querySelector('.gd-packet').style.left=step>=3?'60px':step===2?'30px':'0px';
      scene.querySelector('.gd-packet').style.opacity=step===0?'0':'1';
      scene.querySelector('[data-barrier]').textContent=['descriptor：base + shape + stride + tile shape','线程发出请求：起点 (2,4) → 目标 shared buffer','传输在途：请求已发出，barrier 尚未完成','传输已完成：等待相应 barrier 后，消费者可以读取','消费者使用这 8 个值计算；用完后才能让生产者覆盖'][step];
      return ['先描述矩阵布局。蓝色格子是要搬的 tile；输入值没有经过矩阵乘法。','一个线程提交整块搬运任务，其他线程可以继续做无依赖的工作。','此时不能因为“指令已发出”就读取目标 buffer。','绿色数据已到达，完成同步建立了安全读取的条件。','TMA 负责搬运，Tensor Core 负责后续矩阵计算。'][step];
    }
  });

  mount({
    id:'boundary',section:'11.2',title:'边界位置为什么会浪费整块 tile',
    intro:'小例子 M=N=5。圆点是有效输出，“×”是 tile 覆盖到的越界位置；逐块点亮。',
    options:[option('size','BM=BN',[[4,'4'],[2,'2'],[1,'1']])],
    steps:({options})=>Math.ceil(5/+options.size)**2,
    note:'比例只表示覆盖面积，不是实测 GPU 利用率。小 tile 减少边界空洞，也可能减少数据复用。',
    draw(scene,{step,options}) {
      const b=+options.size,tiles=Math.ceil(5/b),covered=tiles*b;
      scene.innerHTML=`<div class="gd-matrices">${matrix(`D [5,5]，覆盖 ${covered}×${covered}`,covered,covered,(r,c)=>r<5&&c<5?'·':'×',(r,c)=>{
        const id=Math.floor(r/b)*tiles+Math.floor(c/b);
        return `${r>=5||c>=5?'is-pad ':''}${id<step?'is-c ':''}${id===step-1?'is-now':''}`;
      })}</div><div class="gd-formula">${tiles}×${tiles} = ${tiles**2} 个输出 tiles<br>有效面积：25 / ${covered**2} = <strong>${(2500/covered**2).toFixed(1)}%</strong></div>`;
      return `已显示 ${step} / ${tiles**2} 块。有效输出始终只有 25 个，tile 大小改变了边界覆盖量和 CTA 数量。`;
    }
  });

  mount({
    id:'waves',section:'11.3',title:'多出一个 CTA，尾部会发生什么',
    intro:'简化设备有 8 个 SM，每个 SM 一次容纳 1 个 CTA。切换总任务数，逐批观察尾部空闲。',
    options:[option('count','CTA 总数',[[9,'9'],[8,'8'],[16,'16'],[17,'17']])],
    steps:({options})=>Math.ceil(+options.count/8),
    note:'这是等长 CTA、锁步批次的教学模型；真实 GPU 会异步调度，不要求全局按 wave 同步。',
    draw(scene,{step,options}) {
      const count=+options.count,base=(step-1)*8,active=step?Math.min(8,count-base):0;
      scene.innerHTML=`<div class="gd-parts">${seq(8).map(i=>`<div class="gd-part ${i<active?'is-current':''}">SM ${i}<strong>${i<active?`CTA ${base+i}`:'空闲'}</strong></div>`).join('')}</div>
        <div class="gd-formula">${count} 个 CTAs / 8 个并发槽 → ${Math.ceil(count/8)} 批<br>当前批次：${step}；活跃槽：${active} / 8</div>`;
      return step?`第 ${step} 批只有 ${active} 个 CTA。${active<8?'剩余 SM 在这个简化模型中没有工作。':'这一批填满了全部并发槽。'}`:'点击下一步，开始第一批工作。';
    }
  });

  mount({
    id:'split-k',section:'13.',title:'三个 CTA 算同一个输出，最后再归约',
    intro:'仍计算 [1,2,3,4,5,6] 的平方和。每个 CTA 分到两个 K 元素；所有部分和属于同一个 D[0,0]。',
    steps:3,
    draw(scene,{step}) {
      const partial=[5,25,61];
      scene.innerHTML=`<div class="gd-parts">${partial.map((p,i)=>`<div class="gd-part ${step?'is-current':''}">CTA ${i} · K[${i*2},${i*2+2})<strong>${step?`${2*i+1}² + ${2*i+2}² = ${p}`:'等待计算'}</strong>${step>=2?`workspace[${i}] = ${p}`:'独立的部分累加器'}</div>`).join('')}</div>
        <div class="gd-flow"><span class="gd-node ${step>=2?'is-s':''}">workspace<br>${step>=2?'[5, 25, 61]':'[·, ·, ·]'}</span><span class="gd-arrow">→ 归约 →</span><span class="gd-node ${step===3?'is-c':''}">D[0,0]<br><strong>${step===3?'91':'待合并'}</strong></span></div>`;
      return ['三个 CTA 可以独立处理各自的 K 段。','三个部分和可以并行产生；它们并不是三个不同的最终输出。','把三个部分结果写到不同位置，避免无同步地覆盖同一输出。','归约得到 5 + 25 + 61 = 91。bias、βC 和非线性激活应放在正确的合并位置。'][step];
    }
  });

  mount({
    id:'persistent',section:'14.1',title:'同一个 CTA，连续领取不同输出 tiles',
    intro:'9 个输出 tiles，启动 3 个 persistent CTAs。示例使用静态步进：CTA i 依次处理 i、i+3、i+6。',
    steps:3,
    note:'这里只演示 CTA 与工作任务的映射；不表示模型权重一直驻留在 shared memory。',
    draw(scene,{step}) {
      scene.innerHTML=`<div class="gd-parts">${seq(3).map(i=>`<div class="gd-part ${step?'is-current':''}">CTA ${i}<strong>${step?`正在处理 tile ${i+(step-1)*3}`:'等待任务'}</strong>任务序列：${i} → ${i+3} → ${i+6}</div>`).join('')}</div>
        <div class="gd-matrices" style="margin-top:16px">${matrix('9 个独立输出 tiles',3,3,(r,c)=>r*3+c,(r)=>r<step-1?'is-done':r===step-1?'is-c':'')}</div>`;
      return step?`同一批 CTA 的第 ${step} 份任务。绿色一行对应当前工作；前面的行已完成。`:'输出有 9 块，CTA 只有 3 个；每个 CTA 会循环处理多块输出。';
    }
  });

  mount({
    id:'grouped',section:'15.',title:'不同 expert 的 tiles，放进同一组任务',
    intro:'沿用正文的 37、112、9 个 token。只画一个 N tile，BM=32，于是三个 expert 分别有 2、4、1 个 M tiles。',
    steps:7,
    note:'这里按 expert 顺序展示归属；真实 grouped kernel 可以采用其他调度顺序。',
    draw(scene,{step}) {
      const tasks=[[0,0,32],[0,1,5],[1,0,32],[1,1,32],[1,2,32],[1,3,16],[2,0,9]];
      scene.innerHTML=`<div class="gd-parts">${tasks.map(([e,t,n],i)=>`<div class="gd-part ${i===step-1?'is-current':''}">E${e} · tile ${t}<strong>${n} 个有效 token</strong>${i<step?'已分配':'待分配'}</div>`).join('')}</div>
        <div class="gd-formula">${step?`当前任务：X${tasks[step-1][0]} 的第 ${tasks[step-1][1]} 块 × W${tasks[step-1][0]}.T`:'每个任务都携带 expert ID 和 tile 坐标'}</div>`;
      return step?`该 tile 只能读取 expert ${tasks[step-1][0]} 的权重。合并任务分配不意味着把 token 和所有 expert 权重相乘。`:'共有 7 个输出 tiles，来自三个不同大小的 GEMM。';
    }
  });

  mount({
    id:'epilogue',section:'16.',title:'ReLU 融合后，哪次显存往返消失了',
    intro:'GEMM 完成全部 K 累加后，结果为 [-2,3,-1,4]。对比单独 ReLU 与融合 epilogue。',
    steps:3,
    note:'这里只计算 4 个 FP32 输出元素的读写，不含 A/B 输入；步骤并列用于比较数据路径，不代表耗时相同。',
    draw(scene,{step}) {
      scene.innerHTML=`<span class="gd-label">分开执行</span><div class="gd-flow">
        <span class="gd-node ${step>=1?'is-c':''}">完整 acc<br>[-2,3,-1,4]</span><span class="gd-arrow">→</span>
        <span class="gd-node ${step>=2?'is-b':''}">中间结果写入<br>global memory</span><span class="gd-arrow">→</span>
        <span class="gd-node ${step>=3?'is-c':''}">ReLU 读回并写出<br>[0,3,0,4]</span></div>
        <span class="gd-label">融合 epilogue</span><div class="gd-flow">
        <span class="gd-node ${step>=1?'is-c':''}">完整 acc<br>[-2,3,-1,4]</span><span class="gd-arrow">→</span>
        <span class="gd-node ${step>=2?'is-s':''}">片上做 ReLU<br>[0,3,0,4]</span><span class="gd-arrow">→</span>
        <span class="gd-node ${step>=3?'is-c':''}">直接写最终结果<br>global memory</span></div>
        <div class="gd-formula">${step===3?'分开：写 16 B + 读 16 B + 写 16 B = 48 B<br>融合：只写最终结果 16 B':'完整 K 累加 → 后处理 → 最终写回'}</div>`;
      return ['先等待整个 K 归约完成。','两条路径从相同的完整 acc 出发。','融合路径直接在片上处理结果，省去中间矩阵的写回。','数学结果相同；融合减少一次中间写入和一次读回，也可能减少额外 launch。'][step];
    }
  });

  mount({
    id:'scales',section:'17.1',title:'每段点积，要乘上这一段自己的 scales',
    intro:'三个低精度部分点积是 5、25、61。用不同的 scale 乘积，观察缩放后的结果再怎样相加。',
    steps:3,
    draw(scene,{step}) {
      const partial=[5,25,61],scales=[2,1,.5],correct=partial.map((p,i)=>p*scales[i]);
      scene.innerHTML=`<div class="gd-parts">${partial.map((p,i)=>`<div class="gd-part ${i<step?'is-current':''}">K 组 ${i}<strong>P${i} = ${p}</strong>SA × SW = ${scales[i]}<br>贡献：${i<step?`${p} × ${scales[i]} = ${correct[i]}`:'待缩放'}</div>`).join('')}</div>
        <div class="gd-formula">acc = ${step?correct.slice(0,step).join(' + '):'0'} = <strong>${sum(correct.slice(0,step))}</strong></div>`;
      return step===3?'结果是 65.5。三个 scale 乘积不同，不能把未缩放的 91 直接乘上任意一组 scale。':'先处理这一组的部分点积与 scale，再加入同一个累加器。';
    }
  });

  mount({
    id:'scale-layout',section:'17.2',title:'逻辑坐标不变，看 scale 在物理内存中搬家',
    intro:'S 的逻辑形状始终是 [4,3]。绿色标出同一个 K 分组；播放时，下方 12 个元素会从“每行连续”移动到“每个 K 组连续”。',
    options:[option('g','关注 K 分组',[[0,'组 0'],[1,'组 1'],[2,'组 2']])],
    steps:1,
    draw(scene,{step,options}) {
      const g=+options.g;
      if(!scene.firstChild) {
        scene.innerHTML=`<div data-logical></div><span class="gd-label">物理内存：从左到右，地址递增</span>
          <div class="gd-storage">${seq(12).map(i=>`<span class="gd-token" data-token="${i}" style="--slot:${i}">${'abcd'[Math.floor(i/3)]}${i%3}</span>`).join('')}</div><div class="gd-formula" data-stride></div>`;
      }
      scene.querySelector('[data-logical]').innerHTML=`<div class="gd-matrices">${matrix('逻辑 S[m,g] · 值始终不变',4,3,(r,c)=>'abcd'[r]+c,(_,c)=>c===g?'is-c':'')}</div>`;
      scene.querySelectorAll('[data-token]').forEach(token=>{
        const i=+token.dataset.token,r=Math.floor(i/3),c=i%3;
        token.style.setProperty('--slot',step?c*4+r:i);
        token.classList.toggle('is-c',c===g);
      });
      scene.querySelector('[data-stride]').textContent=step?'新 stride（元素单位）= (1,4)；一组的 a/b/c/d scales 连续':'原 stride（元素单位）= (3,1)；一组的 a/b/c/d scales 相隔 3 个元素';
      return step?'这一步表示实际重排数据。逻辑 shape 和每个 S[m,g] 的值没有改变；只有物理地址改变。':'同一轮 K 需要的绿色元素目前分散在内存中。点击“下一步”观察它们移动到一起。';
    }
  });
})();
