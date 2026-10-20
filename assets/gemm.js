/* Page-local teaching figures. Only the pipeline has a clock: spatial mappings
 * stay visible together, and parameter changes recompute a complete example.
 * Counts and timelines are illustrative models, not GPU measurements.
 */
(() => {
  'use strict';
  const article = document.querySelector('article');
  if (!article) return;
  const seq = n => Array.from({length: n}, (_, i) => i);
  const sum = xs => xs.reduce((a, b) => a + b, 0);
  const aValue = (m, k) => (m + k) % 4 + 1;
  const bValue = (k, n) => (k + 2 * n) % 3 + 1;
  const dot = (m, n, end, start = 0) => sum(seq(end - start).map(i => aValue(m, start + i) * bValue(start + i, n)));
  const option = (name, label, values) => ({name, label, values});
  const choices = values => values.map(x => [x, String(x)]);
  const metric = (value, label) => `<div class="gd-metric"><strong>${value}</strong><span>${label}</span></div>`;

  function matrix(label, rows, cols, value, state = () => '', selectable = false) {
    return `<div class="gd-matrix"><span class="gd-label">${label}</span><div class="gd-grid" style="--cols:${cols}" role="group" aria-label="${label}">
      ${seq(rows).map(r => seq(cols).map(c => selectable
        ? `<button type="button" class="gd-cell ${state(r,c)}" data-output="${r},${c}" aria-label="选择 D[${r},${c}]" aria-pressed="${state(r,c) === 'is-c'}">${value(r,c)}</button>`
        : `<span class="gd-cell ${state(r,c)}" aria-label="第 ${r} 行，第 ${c} 列：${value(r,c)}">${value(r,c)}</span>`
      ).join('')).join('')}</div></div>`;
  }

  function mount(config) {
    const heading = [...article.querySelectorAll('h2,h3,h4')].find(h => h.textContent.startsWith(config.section + ' '));
    if (!heading) throw new Error(`Missing GEMM section ${config.section}`);
    const el = document.createElement('figure');
    el.className = `gemm-demo gd-${config.kind || 'static'}`;
    el.id = `gemm-${config.id}`;
    el.setAttribute('aria-labelledby', `${el.id}-title`);
    el.innerHTML = `<figcaption id="${el.id}-title"><span class="gd-kind">${config.label}</span><strong>${config.title}</strong></figcaption>
      <p class="gd-intro">${config.intro}</p>
      ${(config.options || []).length ? `<div class="gd-options">${config.options.map(o => `<label>${o.label}<select data-option="${o.name}" aria-label="${config.title}：${o.label}">${o.values.map(([value,label]) => `<option value="${value}">${label}</option>`).join('')}</select></label>`).join('')}</div>` : ''}
      ${config.kind === 'walk' ? '<div class="gd-controls"><button type="button" data-action="prev">上一段 K</button><output class="gd-counter"></output><button type="button" data-action="next">下一段 K</button><button type="button" data-action="reset">回到初始</button></div>' : ''}
      <div class="gd-scene"></div><p class="gd-status"${config.kind ? ' role="status" aria-live="polite"' : ''}></p>
      ${config.note ? `<p class="gd-note">${config.note}</p>` : ''}`;
    // Replace matching static fallbacks in place; do not stack two figures
    // explaining the same relationship or move a parent figure before its subsections.
    const fallback = config.image && article.querySelector(`img[src="assets/${config.image}"]`);
    if (fallback) fallback.parentElement.replaceWith(el);
    else {
      let next = heading.nextElementSibling;
      const level = +heading.tagName.slice(1);
      while (next && !(/^H[1-6]$/.test(next.tagName) && +next.tagName.slice(1) <= level)) next = next.nextElementSibling;
      article.insertBefore(el, next);
    }
    const state = {step: 0, options: Object.fromEntries((config.options || []).map(o => [o.name, o.values[0][0]]))};
    const scene = el.querySelector('.gd-scene');
    function render() {
      if (config.kind === 'walk') {
        const max = config.steps(state);
        el.querySelector('.gd-counter').textContent = `已累加 ${state.step} / ${max} 段`;
        el.querySelector('[data-action=prev]').disabled = state.step === 0;
        el.querySelector('[data-action=next]').disabled = state.step === max;
      }
      el.querySelector('.gd-status').textContent = config.draw(scene, state);
    }
    el.querySelectorAll('[data-option]').forEach(select => select.addEventListener('change', () => {
      state.options[select.dataset.option] = select.value;
      state.step = 0;
      render();
    }));
    el.addEventListener('click', event => {
      const output = event.target.closest('[data-output]');
      if (output) {
        [state.options.m, state.options.n] = output.dataset.output.split(',').map(Number);
        render();
        el.querySelector(`[data-output="${state.options.m},${state.options.n}"]`).focus({preventScroll: true});
      }
      const button = event.target.closest('[data-action]');
      if (!button) return;
      state.step = button.dataset.action === 'reset' ? 0 : Math.max(0, Math.min(config.steps(state), state.step + (button.dataset.action === 'next' ? 1 : -1)));
      render();
    });
    render();
    return el;
  }

  mount({
    id:'dot', section:'1.', kind:'explore', label:'选点查看', title:'一个输出，对应一整行与一整列',
    intro:'点击 D 中的任意数值，同时查看它使用的 A 行、B 列和完整点积。蓝色对应 A，橙色对应 B，绿色是选中的输出。',
    draw(scene, {options}) {
      const m = options.m || 0, n = options.n || 0;
      scene.innerHTML = `<div class="gd-matrices">${matrix('A [3,6]',3,6,aValue,r=>r===m?'is-a':'')}
        <span class="gd-operator">×</span>${matrix('B [6,3]',6,3,bValue,(_,c)=>c===n?'is-b':'')}
        <span class="gd-operator">=</span>${matrix('D [3,3] · 点击一个输出',3,3,(r,c)=>dot(r,c,6),(r,c)=>r===m&&c===n?'is-c':'',true)}</div>
        <div class="gd-products">${seq(6).map(k=>`<span><small>k = ${k}</small><b class="gd-a-text">${aValue(m,k)}</b> × <b class="gd-b-text">${bValue(k,n)}</b><strong>${aValue(m,k)*bValue(k,n)}</strong></span>`).join('')}</div>
        <div class="gd-formula">D[${m},${n}] = ${seq(6).map(k=>aValue(m,k)*bValue(k,n)).join(' + ')} = <strong>${dot(m,n,6)}</strong></div>`;
      return '换一个输出，换的是行与列；每个输出都要沿完整的 K 求和。';
    }
  });

  // Section 2 already contains a complete six-term worked example; no extra widget.
  mount({
    id:'tiles',section:'4.',kind:'walk',label:'分步推演',title:'输出块固定，输入块沿 K 前进',image:'gemm-tiling.svg',
    intro:'M=N=4、K=6，每个 CTA 负责 2×2 输出。手动推进 K 段，看同一块 acc 怎样累加。',
    options:[option('tile','输出块',[[0,'左上'],[1,'右上'],[2,'左下'],[3,'右下']]),option('bk','每段 BK',choices([2,1,3]))],
    steps:({options})=>6/+options.bk,
    draw(scene,{step,options}) {
      const bk=+options.bk, m0=Math.floor(+options.tile/2)*2, n0=(+options.tile%2)*2;
      const start=Math.max(0,step-1)*bk,end=step*bk;
      scene.innerHTML=`<div class="gd-matrices">
        ${matrix('A [4,6] · 本段 2×BK',4,6,aValue,(r,c)=>r>=m0&&r<m0+2&&c>=start&&c<start+bk?'is-a':'')}
        ${matrix('B [6,4] · 本段 BK×2',6,4,bValue,(r,c)=>c>=n0&&c<n0+2&&r>=start&&r<start+bk?'is-b':'')}
        ${matrix('D [4,4] · 同一块 acc',4,4,(r,c)=>r>=m0&&r<m0+2&&c>=n0&&c<n0+2?dot(r,c,end):'·',(r,c)=>r>=m0&&r<m0+2&&c>=n0&&c<n0+2?'is-c':'')}
        </div><div class="gd-metrics">${metric('4','独立输出 tiles / CTAs')}${metric(6/bk,'每个 CTA 内的 K 段数')}${metric(`[${m0},${n0}]`,'当前输出块的起点')}</div>`;
      return step ? `刚处理 K=[${start},${end})。${end===6?'这一块输出已完成。':'下一段继续更新相同的四个输出。'}` : '高亮的是即将使用的第一段输入；acc 从零开始。';
    }
  });

  mount({
    id:'reuse',section:'5.',kind:'explore',label:'复用计算器',title:'输出 tile 变大，同一份输入能用几次',
    intro:'固定 BK=32。改变输出块的行数和列数，直接比较两种加载组织的元素请求数。',
    options:[option('bm','BM',choices([32,64,128])),option('bn','BN',choices([32,64,128]))],
    note:'仅统计一段 K 的输入元素请求；忽略 cache、输出和额外状态，不能把比值当成实际带宽或加速比。',
    draw(scene,{options}) {
      const m=+options.bm,n=+options.bn,independent=2*m*n*32,cooperative=(m+n)*32,ratio=independent/cooperative;
      scene.innerHTML=`<div class="gd-reuse"><div><span class="gd-a-text">A tile：${m}×32</span><strong>每个值用于 ${n} 个输出列</strong></div><div><span class="gd-b-text">B tile：32×${n}</span><strong>每个值用于 ${m} 个输出行</strong></div></div>
        <div class="gd-compare-bars"><div><span>逐输出独立请求</span><div class="gd-bar-track"><b class="is-b" style="width:100%"></b></div><strong>${independent.toLocaleString()}</strong></div>
        <div><span>协作加载后复用</span><div class="gd-bar-track"><b class="is-c" style="width:${cooperative/independent*100}%"></b></div><strong>${cooperative.toLocaleString()}</strong></div></div>
        <div class="gd-formula">独立请求：2 × BM × BN × BK<br>协作加载：(BM + BN) × BK<br>本例请求数之比：<strong>${Number.isInteger(ratio)?ratio:'≈ '+ratio.toFixed(2)} : 1</strong></div>`;
      return `复用增加的同时，要保存的 FP32 acc 也增至 ${m*n} 个值（${m*n*4/1024} KiB）。更大的 tile 有资源代价。`;
    }
  });

  mount({
    id:'registers',section:'6.',label:'外积展开',title:'两个 A 值、两个 B 值，更新四个寄存器',
    intro:'只展开固定 k 的一次更新。A 值沿行复用，B 值沿列复用；四个 acc 都保留原来的部分和。',
    draw(scene) {
      scene.innerHTML=`<div class="gd-outer"><span></span><b class="is-b">b₀</b><b class="is-b">b₁</b>
        <b class="is-a">a₀</b><span class="is-c">acc₀₀ += a₀b₀</span><span class="is-c">acc₀₁ += a₀b₁</span>
        <b class="is-a">a₁</b><span class="is-c">acc₁₀ += a₁b₀</span><span class="is-c">acc₁₁ += a₁b₁</span></div>`;
      return '这是一个线程负责 2×2 输出的示例：下一轮 k 更换 a/b，继续更新这四个 acc。';
    }
  });

  mount({
    id:'tmem',section:'7.3',label:'存储与数据流',title:'每段更换输入，累加结果留在 TMEM',
    intro:'SM100 的一种数据流：A/B 都从 shared memory 提供。箭头表示数据流，不表示各阶段耗时。',
    draw(scene) {
      scene.innerHTML=`<div class="gd-tmem-map"><div class="gd-storage-lane"><span class="gd-lane-title">输入 · 逐段更换</span>
        <div class="gd-node is-a">Shared A<small>A₀ → A₁ → A₂</small></div><div class="gd-node is-b">Shared B<small>B₀ → B₁ → B₂</small></div></div>
        <div class="gd-link-down">↓ 矩阵块交给 Tensor Core</div>
        <div class="gd-accumulator is-c"><strong>同一块 TMEM acc</strong><span>acc ← acc + Aₖ @ Bₖ</span><small>沿 K 持续更新；部分结果留在这里</small></div>
        <div class="gd-link-down">↓ 全部 MMA 完成后，读取结果并等待完成</div>
        <div class="gd-output-path"><span>普通寄存器<br><small>bias / 激活 / 类型转换</small></span><b aria-hidden="true">→</b><span>Global memory<br><small>最终输出</small></span></div></div>`;
      return 'TMA 搬运输入，Tensor Core 做矩阵运算，TMEM 保存累加数据；三者的职责不同。';
    }
  });

  mount({
    id:'coalescing',section:'8.1',label:'地址对照',title:'同一条访存指令，看全部 lanes 的地址集合',
    intro:'仅画前 8 个 lanes。row-major FP32 每行 8 个元素，起点按 32 bytes 对齐；每条横带是一个 32-byte sector。',
    note:'8 个 lanes 的简化示意，不把它当成完整 warp。覆盖 sector 数也不等于实际 DRAM 事务数。',
    draw(scene) {
      scene.innerHTML=`<div class="gd-comparison">${[false,true].map(column=>`<div><h4>${column?'跨行读一列':'同一行相邻元素'}</h4><p class="gd-mini">字节偏移 = lane × ${column?32:4}</p>
        <div class="gd-sectors">${seq(8).map(s=>`<div class="gd-sector"><small>${s*32}–${s*32+31} B</small><div>${seq(8).map(c=>`<span class="${column?c===0?'is-b':'':s===0?'is-a':''}">${column?c===0?`L${s}`:'·':s===0?`L${c}`:'·'}</span>`).join('')}</div></div>`).join('')}</div>
        <p class="gd-result">${column?'8 个 sectors，每个取 4 B':'1 个 sector，取连续 32 B'}</p></div>`).join('')}</div>`;
      return '两边都只请求 8 个 FP32 值。区别在地址分布，不在线程先后顺序。';
    }
  });

  mount({
    id:'banks',section:'8.2.2',label:'映射对照',title:'padding 改变地址到 bank 的映射',
    intro:'同一条读取指令，lane i 取第 i 行的第 0 列。FP32、32 banks、起始 bank 0；下方同时显示三种结果。',
    draw(scene) {
      scene.innerHTML=`<div class="gd-bank-cases">${[
        ['stride = 32','bank = (lane × 32) % 32','不同地址集中到 bank 0','32-way conflict','is-b'],
        ['stride = 33','bank = (lane × 33) % 32','bank 与 lane 一一对应','无 bank conflict','is-c'],
        ['同一地址','所有 lanes 都读取 x[0]','一个地址，供所有 lanes 使用','广播','is-a']
      ].map(([name,formula,caption,result,color],mode)=>`<div class="gd-bank-case"><div><strong>${name}</strong><code>${formula}</code></div>
        <div class="gd-bank-loads" aria-label="${caption}">${seq(32).map(b=>`<span class="${mode===1||b===0?color:''}" title="bank ${b}：${mode===1?'1 个地址':b===0?mode===0?'32 个不同地址':'1 个地址广播':'无访问'}"><b>${mode===1?'1':b===0?mode===0?'32':'1':'·'}</b><small>${b%8===0?b:''}</small></span>`).join('')}</div>
        <p><b>${result}</b> · ${caption}</p></div>`).join('')}</div>
        <table class="gd-table"><thead><tr><th>lane</th><th>stride 32 → bank</th><th>stride 33 → bank</th><th>读取同一地址</th></tr></thead><tbody>${[0,1,2,31].map(i=>`<tr><th>${i}</th><td>${i*32} → 0</td><td>${i*33} → ${i}</td><td>0 → 0</td></tr>`).join('')}</tbody></table>`;
      return '依次排列 bank 0–31，格内数字是本次访存涉及的不同地址数。“32”集中在一个 bank 与“1”分散到各 bank，服务方式不同；同一地址可以广播。';
    }
  });

  const serialLoads=seq(4).map(i=>[i*5,2,i]);
  const serialComputes=seq(4).map(i=>[i*5+2,3,i]);
  const overlapLoads=[[0,2,0],[2,2,1],[5,2,2],[8,2,3]];
  const overlapComputes=seq(4).map(i=>[2+i*3,3,i]);
  function track(label,events,type) {
    return `<div class="gd-track"><span>${label}</span><div class="gd-rail">${events.map(([start,duration,id])=>`<span class="gd-bar is-${type==='L'?'a':'c'}" style="left:${start*5}%;width:${duration*5}%">${type}${id}</span>`).join('')}<span class="gd-cursor"></span></div></div>`;
  }
  const pipeline=mount({
    id:'pipeline',section:'9.',kind:'timeline',label:'时间轴',title:'双缓冲：何时能读，何时能覆盖',image:'gemm-pipeline.svg',
    intro:'L 是加载，C 是计算，下标是 K tile。完整时间轴始终可见；拖动时间游标或播放，查看两块 buffer 的状态。',
    note:'假设加载 2 个时间单位、计算 3 个；这是理想化依赖模型，不是 GPU 实测时间。',
    draw(scene) {
      scene.innerHTML=`<div class="gd-timeline">${track('串行 · 加载',serialLoads,'L')}${track('串行 · 计算',serialComputes,'C')}<div class="gd-track-gap"></div>${track('双缓冲 · 加载',overlapLoads,'L')}${track('双缓冲 · 计算',overlapComputes,'C')}
        <div class="gd-track gd-axis"><span>时间 →</span><div>${[0,5,10,15,20].map(t=>`<span style="left:${t*5}%">${t}</span>`).join('')}</div></div></div>
        <div class="gd-timeline-controls"><button type="button" class="gd-play" aria-pressed="false">播放时间轴</button><label>时间 t<input class="gd-time" type="range" min="0" max="20" value="5" step="1" aria-label="流水线时间"></label><output class="gd-time-value">5</output></div><div class="gd-slots"></div>`;
      return 't=5：tile 0 用完，buffer 0 可以装 tile 2；buffer 1 中的 tile 1 开始计算。';
    }
  });
  let time=5, timer=null;
  const timeInput=pipeline.querySelector('.gd-time'), play=pipeline.querySelector('.gd-play');
  function pause() {
    clearInterval(timer);timer=null;
    play.textContent='播放时间轴';play.setAttribute('aria-pressed','false');
  }
  function renderTime() {
    timeInput.value=time;
    pipeline.querySelector('.gd-time-value').textContent=time;
    pipeline.querySelectorAll('.gd-cursor').forEach(cursor=>{cursor.style.left=`${time*5}%`;});
    pipeline.querySelector('.gd-slots').innerHTML=seq(2).map(slot=>{
      const loads=overlapLoads.filter(e=>e[2]%2===slot&&e[0]<=time),e=loads[loads.length-1];
      let label='空闲',kind='';
      if(e) {
        const [start,duration,id]=e,compute=overlapComputes[id];
        if(time<start+duration){label=`tile ${id} 加载中 · 不可读`;kind='is-a';}
        else if(time<compute[0]){label=`tile ${id} 就绪 · 等待计算`;kind='is-s';}
        else if(time<compute[0]+compute[1]){label=`tile ${id} 计算中 · 不可覆盖`;kind='is-c';}
        else label=`tile ${id} 用完 · 可以覆盖`;
      }
      return `<div class="gd-slot ${kind}"><b>buffer ${slot}</b><span>${label}</span></div>`;
    }).join('');
    pipeline.querySelector('.gd-status').textContent=`t=${time}。${time<2?'等待第一块输入。':time<14?'搬运和计算使用不同缓冲槽；上次计算读完后，槽才能复用。':time<20?'双缓冲已在 t=14 完成；串行路径仍在工作。':'串行在 t=20 完成。两条路径处理了相同的四块输入。'}`;
  }
  timeInput.addEventListener('input',()=>{pause();time=+timeInput.value;renderTime();});
  play.addEventListener('click',()=>{
    if(timer!==null){pause();return;}
    if(time>=20)time=0;
    play.textContent='暂停';play.setAttribute('aria-pressed','true');
    renderTime();
    timer=setInterval(()=>{time++;renderTime();if(time===20)pause();},750);
  });
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)pause();}).observe(pipeline);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
  renderTime();

  mount({
    id:'tma',section:'10.1',label:'职责与依赖',title:'一次搬运请求，两条不同的路径',
    intro:'把“线程提交请求”和“数据真正到达”分开看。实线箭头表示数据流，虚线箭头表示请求或完成通知。',
    draw(scene) {
      scene.innerHTML=`<div class="gd-tma-map"><div class="gd-tma-control"><b>发起线程</b><span>descriptor + tile 坐标</span><span class="gd-dashed-down">↓ 提交请求</span></div>
        <div class="gd-tma-data"><div class="gd-node">Global memory<br><small>源矩阵</small></div><b aria-hidden="true">→</b><div class="gd-node is-a">TMA<br><small>寻址与搬运</small></div><b aria-hidden="true">→</b><div class="gd-node is-c">Shared memory<br><small>目标 buffer</small></div></div>
        <div class="gd-tma-completion"><span class="gd-dashed-down">↓ 完成通知</span><b>关联的 barrier</b><span>消费者等待完成 → 读取 buffer</span></div></div>`;
      return '请求返回后，数据可能还在途中。消费者必须等搬运完成；消费结束后，生产者才能覆盖这个 buffer。';
    }
  });

  mount({
    id:'boundary',section:'11.2',kind:'explore',label:'边界计算器',title:'同样 25 个输出，不同 tile 覆盖多少空位',
    intro:'D 是 5×5。粗边界划分输出 tiles；点表示有效输出，斜线区域表示越界覆盖。所有 tiles 同时显示。',
    options:[option('size','BM=BN',choices([4,2,1]))],
    note:'覆盖比例不是实测 GPU 利用率。这里只改变输出 tile，未模拟 occupancy、访存或指令执行。',
    draw(scene,{options}) {
      const b=+options.size,tiles=Math.ceil(5/b),covered=tiles*b;
      scene.innerHTML=`<div class="gd-boundary-layout"><div class="gd-tile-grid" style="--tiles:${tiles}" aria-label="${tiles*tiles} 个输出 tiles">${seq(tiles*tiles).map(i=>`<div class="gd-output-tile" style="--cols:${b}">${seq(b*b).map(j=>{const r=Math.floor(i/tiles)*b+Math.floor(j/b),c=(i%tiles)*b+j%b;return `<span class="${r<5&&c<5?'is-c':'is-pad'}">${r<5&&c<5?'·':'×'}</span>`;}).join('')}</div>`).join('')}</div>
        <div class="gd-metrics gd-metrics-stack">${metric(tiles*tiles,'输出 tiles / 基础分工 CTA 数')}${metric(covered*covered-25,'越界覆盖的位置')}${metric(`${(2500/(covered*covered)).toFixed(1)}%`,'有效面积 / 总覆盖面积')}</div></div>`;
      return `有效输出始终是 25 个。BM=BN=${b} 时，覆盖 ${covered}×${covered} 个位置；减小 tile 也会改变数据复用和任务数量。`;
    }
  });

  mount({
    id:'waves',section:'11.3',kind:'explore',label:'排布计算器',title:'把所有批次同时摆出来，看尾部空槽',
    intro:'简化设备有 8 个 SM，每个 SM 同时运行一个等长 CTA。横向看并发槽，纵向看先后批次。',
    options:[option('count','CTA 数量',choices([9,8,16,17]))],
    note:'真实 GPU 异步调度，不需要批次之间全局同步。这里用等长、锁步模型解释尾部效应。',
    draw(scene,{options}) {
      const count=+options.count,waves=Math.ceil(count/8);
      scene.innerHTML=`<div class="gd-wave-grid"><span></span>${seq(8).map(i=>`<small>SM ${i}</small>`).join('')}${seq(waves).map(w=>`<b>第 ${w+1} 批</b>${seq(8).map(s=>`<span class="${w*8+s<count?'is-c':'is-pad'}">${w*8+s<count?w*8+s:'—'}</span>`).join('')}`).join('')}</div>
        <div class="gd-metrics">${metric(waves,'批次')}${metric(`${count-(waves-1)*8} / 8`,'末批有工作的位置')}${metric(`${(count/(waves*8)*100).toFixed(1)}%`,'等长模型的槽位填充率')}</div>`;
      return '格内数字为 CTA 编号。8 → 9 或 16 → 17 时，只多一个任务，却需要额外一批。';
    }
  });

  mount({
    id:'split-k',section:'13.',label:'分叉与归约',title:'三个并行部分和，汇入同一个输出',
    intro:'用 [1,2,3,4,5,6] 的平方和举例。沿 K 分给不同 CTA，先写独立位置，再归约。',
    draw(scene) {
      scene.innerHTML=`<div class="gd-split"><div class="gd-split-input">同一个输出 D[0,0]，完整 K = [0,6)</div><div class="gd-branches">${[5,25,61].map((p,i)=>`<div><span class="gd-branch-line" aria-hidden="true">↓</span><div class="gd-node is-a"><b>CTA ${i}</b><small>K=[${2*i},${2*i+2})</small><strong>${2*i+1}² + ${2*i+2}² = ${p}</strong></div><span class="gd-branch-line" aria-hidden="true">↓</span><div class="gd-node is-s">workspace[${i}] = ${p}</div></div>`).join('')}</div>
        <div class="gd-merge-line" aria-hidden="true"></div><div class="gd-split-output is-c">归约：5 + 25 + 61 = <strong>91</strong><small>得到 D[0,0]，再做适当的后处理</small></div></div>`;
      return '三个 CTA 的结果属于同一个输出。并行度增加，同时多了部分结果存储与合并。';
    }
  });

  mount({
    id:'persistent',section:'14.1',label:'任务归属',title:'9 个输出 tiles，分给 3 个常驻 CTA',
    intro:'静态步进示例：CTA i 依次处理 i、i+3、i+6。每条横行是一份 CTA 的任务序列。',
    draw(scene) {
      scene.innerHTML=`<div class="gd-assignment">${seq(3).map(i=>`<div><b>CTA ${i}</b>${seq(3).map(j=>`${j?'<span aria-hidden="true">→</span>':''}<span class="gd-task is-${['a','b','c'][i]}">tile ${i+3*j}</span>`).join('')}</div>`).join('')}</div>`;
      return '箭头表示同一 CTA 内的处理顺序；不同 CTA 可以并行，不表示三行必须同步前进。';
    }
  });

  mount({
    id:'grouped',section:'15.',label:'按规模排布',title:'expert 的 token 数不同，任务仍保留归属',
    intro:'BM=32，只看一个 N tile。每条横带对应一个 expert；格宽表示有效 token 数，斜线补足末尾 tile。',
    draw(scene) {
      const lengths=[37,112,9];
      scene.innerHTML=`<div class="gd-experts">${lengths.map((n,e)=>`<div><span><b>Expert ${e}</b><small>${n} tokens · ${Math.ceil(n/32)} tiles</small></span><div class="gd-expert-track">${seq(Math.ceil(n/32)).map(t=>{const valid=Math.min(32,n-t*32);return `<div class="gd-expert-tile is-pad"><span class="is-${['a','b','c'][e]}" style="width:${valid/32*100}%"></span><b>${valid}</b></div>`;}).join('')}</div><code>× W${e}.T</code></div>`).join('')}</div>
        <p class="gd-result">2 + 4 + 1 = 7 个任务，每个任务携带 expert ID 和 tile 坐标</p>`;
      return 'Grouped GEMM 把不同规模的矩阵问题放到一起调度；每个 tile 仍使用所属 expert 的权重。';
    }
  });

  mount({
    id:'epilogue',section:'16.',label:'数据路径对照',title:'融合 epilogue，省掉一次中间写入和读回',
    intro:'完整 acc = [-2,3,-1,4]，最终 ReLU 输出都是 [0,3,0,4]。直接比较片上处理与 global memory 往返。',
    note:'只统计四个 FP32 输出元素的请求字节数；不含 A/B，不代表实测 DRAM 流量或执行耗时。',
    draw(scene) {
      scene.innerHTML=`<div class="gd-epilogue"><div><b>分开执行</b><ol><li>完整 acc</li><li class="is-b">写中间矩阵 · 16 B</li><li class="is-b">ReLU 读回 · 16 B</li><li>写最终输出 · 16 B</li></ol><strong>48 B</strong></div>
        <div><b>融合后处理</b><ol><li>完整 acc</li><li class="is-c">片上执行 ReLU</li><li>写最终输出 · 16 B</li></ol><strong>16 B</strong></div></div>`;
      return '两条路径都要等完整 K 归约完成。融合路径在写出之前处理结果，省去中间矩阵往返。';
    }
  });

  mount({
    id:'scales',section:'17.1',label:'逐项核算',title:'各 K 组先应用自己的 scale，再相加',
    intro:'低精度部分点积为 5、25、61；三组的 scale 乘积分别为 2、1、0.5。完整计算同时展开。',
    draw(scene) {
      scene.innerHTML=`<table class="gd-table"><thead><tr><th>K 组</th><th>部分点积 P</th><th>SA × SW</th><th>对最终输出的贡献</th></tr></thead><tbody>${[5,25,61].map((p,i)=>`<tr><th>${i}</th><td>${p}</td><td>${[2,1,.5][i]}</td><td class="is-c">${p} × ${[2,1,.5][i]} = <b>${p*[2,1,.5][i]}</b></td></tr>`).join('')}</tbody></table>
        <div class="gd-formula">输出 = 10 + 25 + 30.5 = <strong>65.5</strong></div>`;
      return '三个 scale 乘积不同，不能把未缩放的总和 91 直接乘上其中任意一组 scale。';
    }
  });

  mount({
    id:'scale-layout',section:'17.2',kind:'explore',label:'布局对照',title:'同一组 scale，在两种布局里位于哪里',image:'gemm-scale-layout.svg',
    intro:'S 的逻辑 shape 始终是 [4,3]，a/b/c/d 对应四行。切换 K 分组，比较同一组四个值的物理位置。',
    options:[option('g','关注 K 分组',[[0,'组 0'],[1,'组 1'],[2,'组 2']])],
    note:'比较的是实际重排前后；省略 padding、packing 和硬件对齐要求。',
    draw(scene,{options}) {
      const g=+options.g;
      scene.innerHTML=`<div class="gd-layout-rows">${[false,true].map(reordered=>`<div><b>${reordered?'M 连续 · 重排后':'K 连续 · 原布局'}</b><code>stride = ${reordered?'(1,4)':'(3,1)'}</code><div class="gd-storage">${seq(12).map(slot=>{const r=reordered?slot%4:Math.floor(slot/3),c=reordered?Math.floor(slot/4):slot%3;return `<span class="${c===g?'is-c':''}"><b>${'abcd'[r]}${c}</b><small>${slot}</small></span>`;}).join('')}</div></div>`).join('')}</div>
        <div class="gd-formula">同一组 ${g} 的地址偏移：<br>原布局：${seq(4).map(r=>r*3+g).join('，')}<br>重排后：${seq(4).map(r=>g*4+r).join('，')}<br><small>偏移与 stride 均以元素为单位；下排灰色数字是物理偏移。</small></div>`;
      return 'shape 和每个 S[m,g] 的值保持不变，改变的是数据存放顺序。两种布局同时可见，无需记住上一帧。';
    }
  });
})();
