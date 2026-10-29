/* Page-local teaching models. Matrix colors describe ownership, not physical
 * register layouts. Pipeline durations are abstract stages, never GPU timings.
 * Controls are manual so motion and reading never advance without the reader.
 */
(() => {
  'use strict';
  const seq = n => Array.from({length:n}, (_, i) => i);
  const fmt = n => Number.isInteger(n) ? String(n) : n.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  const metric = (value, label) => `<div class="fa-metric"><strong>${value}</strong><span>${label}</span></div>`;
  const controls = (noun, resetLabel = '回到开始') => `<div class="fa-controls"><button type="button" data-action="prev">上一个${noun}</button><output class="fa-counter"></output><button type="button" data-action="next">下一个${noun}</button><button type="button" data-action="reset">${resetLabel}</button></div>`;

  function mount(id, title, intro, form, note) {
    const el = document.getElementById(id);
    if (!el) return null;
    el.setAttribute('role', 'group');
    el.setAttribute('aria-labelledby', `${id}-title`);
    el.innerHTML = `<div class="fa-caption"><span>动手看</span><strong id="${id}-title">${title}</strong></div><p class="fa-intro">${intro}</p>${form}<div class="fa-scene"></div><p class="fa-status" role="status" aria-live="polite"></p><p class="fa-note">${note}</p>`;
    return el;
  }

  function grid(rows, cols, cell, rowPrefix = 'Q', colPrefix = 'KV') {
    return `<div class="fa-matrix" style="--cols:${cols}" role="img" aria-label="${rows} 行 query、${cols} 列 key 的逻辑矩阵"><span></span>${seq(cols).map(c=>`<span class="fa-axis">${colPrefix}${c}</span>`).join('')}${seq(rows).map(r=>`<span class="fa-axis">${rowPrefix}${r}</span>${seq(cols).map(c=>cell(r,c)).join('')}`).join('')}</div>`;
  }

  const io = mount('fa-io', '完整矩阵写回，还是一块算完就用掉？',
    '下面的 8×8 网格只表示 score 的逻辑坐标。分块模式逐个处理 2×2 区域，不把整张 score 存进 HBM。',
    '<div class="fa-options"><label>计算方式<select data-option="mode"><option value="tiled">FlashAttention：分块使用</option><option value="dense">朴素路径：完整写回</option></select></label></div>' + controls('块'),
    '网格固定一组 Q 行、遍历 KV，演示分块思想，不代表 FA1 原论文的精确循环顺序。尺寸卡片另取单 head、N=8192，只比较 FP32 score 暂存，不是总显存、HBM 流量或加速比。');
  if (io) {
    let step=0;
    const render=()=>{
      const tiled=io.querySelector('select').value==='tiled';
      const qr=Math.floor(step/4)*2, kc=(step%4)*2;
      io.querySelector('.fa-counter').textContent=tiled?`第 ${step+1} / 16 块`:'完整的 8×8 score';
      io.querySelector('[data-action=prev]').disabled=!tiled||step===0;
      io.querySelector('[data-action=next]').disabled=!tiled||step===15;
      io.querySelector('[data-action=reset]').disabled=!tiled||step===0;
      io.querySelector('.fa-scene').innerHTML=`<div class="fa-two"><div>${grid(8,8,(r,c)=>{
        const tile=Math.floor(r/2)*4+Math.floor(c/2);
        const selected=tiled&&r>=qr&&r<qr+2&&c>=kc&&c<kc+2;
        return `<span class="fa-cell ${!tiled?'fa-orange':selected?'fa-blue':tile<step?'fa-consumed':''}">${!tiled?'存':selected?'算':tile<step?'✓':'·'}</span>`;
      })}<p class="fa-legend">${tiled?'蓝色＝当前块；✓＝已经消费，符号不表示仍保存着 score。':'橙色＝写入 HBM 的完整中间矩阵。'}</p></div><div class="fa-storage"><div><span class="fa-tag">HBM</span><strong>${tiled?'Q / K / V，以及最终 O':'Q / K / V、完整 S / P，以及 O'}</strong><p>${tiled?'后续块继续从输入取数。':'写 S → 读 S → 写 P → 读 P。'}</p></div><div><span class="fa-tag fa-tag-green">片上处理</span><strong>${tiled?'当前 score tile + 行状态 m / ℓ / u':'分阶段计算 QK、softmax、PV'}</strong><p>${tiled?'当前块用完即可复用空间；行状态保留。':'完整 score / probability 在阶段间经过 HBM。'}</p></div></div></div><div class="fa-metrics">${metric('256 MiB','完整 S：8192 × 8192 × 4 B')}${metric('32 KiB','score tile：64 × 128 × 4 B')}</div>`;
      io.querySelector('.fa-status').textContent=tiled?`当前 Q 行 [${qr}, ${qr+2})，KV 列 [${kc}, ${kc+2})。同一组 Q 行遍历各 KV 块，更新同一组输出状态。`:'这个视图强调中间结果的存放位置；完整矩阵中的所有允许位置仍然要计算。';
    };
    io.querySelector('select').addEventListener('change',()=>{step=0;render();});
    io.addEventListener('click',e=>{const a=e.target.closest('[data-action]');if(!a)return;step=a.dataset.action==='reset'?0:Math.max(0,Math.min(15,step+(a.dataset.action==='next'?1:-1)));render();});
    render();
  }

  const online = mount('fa-online', '分块后，softmax 的分母怎么接起来？',
    '只看一个 query：scores = [ln 1, ln 2, ln 3, ln 4]，V = [10, 20, 30, 40]。每块两个元素；整行结果应为 30。',
    '<div class="fa-options"><label>参考值策略<select data-option="reference"><option value="max">每块更新最大值</option><option value="keep">第二块保留旧参考值</option></select></label></div>' + controls('阶段','回到初始状态'),
    '这里用未归一化分子 u 演示 online softmax。保留旧参考值只演示 FA4 的缩放原理，不模拟真实 kernel 的阈值、精度或分工。');
  if (online) {
    let step=1;
    const weights=[1,2,3,4], values=[10,20,30,40];
    const render=()=>{
      const keep=online.querySelector('select').value==='keep';
      let c=-Infinity,l=0,u=0,previous=null,alpha=0,added=[];
      for(let block=0;block<step;block++){
        previous={c,l,u};
        const end=2*(block+1),nextC=keep&&block>0?c:Math.log(weights[end-1]);
        alpha=block===0?0:Math.exp(c-nextC);
        added=weights.slice(end-2,end).map(w=>Math.exp(Math.log(w)-nextC));
        l=alpha*l+added.reduce((a,b)=>a+b,0);
        u=alpha*u+added.reduce((a,b,j)=>a+b*values[end-2+j],0);
        c=nextC;
      }
      online.querySelector('.fa-counter').textContent=`已处理 ${step} / 2 块`;
      online.querySelector('[data-action=prev]').disabled=step===0;
      online.querySelector('[data-action=next]').disabled=step===2;
      online.querySelector('[data-action=reset]').disabled=step===0;
      const currentStart=(step-1)*2;
      const tiles=weights.map((w,i)=>`<div class="fa-token ${step&&i>=currentStart&&i<step*2?'fa-blue':i<step*2?'fa-consumed':''}"><span>score = ln ${w}</span><strong>V = ${values[i]}</strong><small>${i<step*2?'已处理':'尚未处理'}</small></div>`).join('');
      const calc=step?`<div class="fa-equation"><strong>旧状态先统一到同一尺度</strong><p>α = ${step===1?'0（没有旧数据）':`exp(${fmt(previous.c)} − ${fmt(c)}) ≈ ${fmt(alpha)}`}</p><p>ℓ = ${fmt(previous.l)} × ${fmt(alpha)} + ${added.map(fmt).join(' + ')} = <b>${fmt(l)}</b></p><p>u = ${fmt(previous.u)} × ${fmt(alpha)} + ${added.map((p,j)=>`${fmt(p)} × ${values[currentStart+j]}`).join(' + ')} = <b>${fmt(u)}</b></p></div>`:'<div class="fa-equation">初始状态：参考值 −∞，ℓ = 0，u = 0。还没有数据，不执行 0 / 0。</div>';
      online.querySelector('.fa-scene').innerHTML=`<div class="fa-tokens">${tiles}</div><div class="fa-metrics">${metric(step?`${keep&&step===2?'ln 2':'ln '+weights[step*2-1]}`:'−∞','参考值 c')}${metric(fmt(l),'分母 ℓ')}${metric(fmt(u),'分子 u')}${metric(step?fmt(u/l):'—',step===2?'最终输出 u / ℓ':'当前前缀 u / ℓ')}</div>${calc}`;
      online.querySelector('.fa-status').textContent=step===0?'先处理第一块，建立第一个参考值。':step===1?'第一块的参考值为 ln 2。此时只是前两个元素的结果；继续加入第二块。':keep?'参考值仍为 ln 2：旧状态不用缩放，新权重为 1.5 和 2。150 / 5 = 30。':'参考值升到 ln 4：旧分子和分母都乘 0.5。75 / 2.5 = 30。';
    };
    online.querySelector('select').addEventListener('change',render);
    online.addEventListener('click',e=>{const a=e.target.closest('[data-action]');if(!a)return;step=a.dataset.action==='reset'?0:Math.max(0,Math.min(2,step+(a.dataset.action==='next'?1:-1)));render();});
    render();
  }

  const work = mount('fa-work', '让不同 warp 拥有不同的输出行',
    '这里用 4 个 query 行、4 个 KV 位置，画一个 CTA 内的抽象分工。W0–W3 表示四个 warp，颜色旁同时保留编号。',
    '<div class="fa-options"><label>CTA 内怎样分<select data-option="partition"><option value="q">sliced-Q：按 Q 行分</option><option value="k">sliced-K：按 KV 列分</option></select></label></div>',
    '图中比较工作归属，不表示 Tensor Core 的精确 lane/register 布局。sliced-Q 减少输出合并，不会取消加载和缓冲区复用所需的同步。');
  if (work) {
    const render=()=>{
      const slicedQ=work.querySelector('[data-option=partition]').value==='q';
      work.querySelector('.fa-scene').innerHTML=`<div class="fa-two"><div>${grid(4,4,(r,c)=>`<span class="fa-cell fa-warp-${slicedQ?r:c}">W${slicedQ?r:c}</span>`)}<p class="fa-legend">列是不同 KV 位置，不是 GEMM 的特征归约轴 d。</p></div><div class="fa-ownership">${seq(4).map(r=>`<div class="fa-output-row">${slicedQ?`<span class="fa-chip fa-warp-${r}">W${r}</span><span>→</span><strong>O 的第 ${r} 行</strong>`:`<div class="fa-contributors">${seq(4).map(w=>`<span class="fa-chip fa-warp-${w}">W${w}</span>`).join('')}</div><span>→ 合并 →</span><strong>O${r}</strong>`}</div>`).join('')}<p class="fa-legend">${slicedQ?'每个 warp 的输出行独立。':'同一输出行的贡献来自多个 warp；需正确合并贡献与 softmax 状态。'}</p></div></div><div class="fa-cta"><strong>再看另一个层次：不同 CTA 之间</strong><p>固定 B=1、H=8、每个 Q tile 有 128 行。</p><label>序列长度 N<select data-option="length"><option value="1024">1024</option><option value="8192">8192</option></select></label><div class="fa-metrics fa-cta-metrics"></div><p class="fa-note">增加 query-tile 并行与上方的 warp 分工是两项独立改进；工作单位数量不等于同时驻留数量。</p></div>`;
      const count=()=>{
        const n=+work.querySelector('[data-option=length]').value;
        work.querySelector('.fa-cta-metrics').innerHTML=metric('8','只按 batch × head 分')+metric(String(8*n/128),`再按 Q tile 分：8 × ${n/128}`);
      };
      work.querySelector('[data-option=length]').value=work.dataset.length||'8192';
      work.querySelector('[data-option=length]').addEventListener('change',e=>{work.dataset.length=e.target.value;count();});count();
      work.querySelector('.fa-status').textContent=slicedQ?'同一 CTA 内按 Q 分工：共同使用 K/V，各自保留自己负责的输出行。':'同一 CTA 内按 KV 分工：多个 warp 为同一输出行提供贡献，需要交换并合并中间结果。';
    };
    work.querySelector('[data-option=partition]').addEventListener('change',render);render();
  }

  const pipeline=mount('fa-pipeline','搬运、矩阵计算、softmax 怎样重叠？',
    'A、B 是两份独立的 tile 工作。同一份工作始终按加载 → QK → softmax → PV 的依赖执行。拖动时间，查看当前正在做什么。',
    '<div class="fa-options"><label>调度方式<select data-option="schedule"><option value="overlap">不同工作交错</option><option value="serial">逐份串行处理</option></select></label></div><div class="fa-controls"><button type="button" data-action="prev">前一步</button><label class="fa-time-control">抽象阶段 t <input type="range" min="0" max="16" step="1" value="6" aria-label="流水线抽象阶段"></label><output class="fa-counter"></output><button type="button" data-action="next">后一步</button><button type="button" data-action="reset">回到开始</button></div>',
    '每种操作在此人为设为 2 个时间单位，仅用于检查依赖和资源重叠；不是 GPU 实测时间，不代表固定加速比或实际 kernel 的精确调度。');
  if(pipeline){
    let time=6;
    const serial=[['load','A',0,2],['mma','A QK',2,4],['soft','A',4,6],['mma','A PV',6,8],['load','B',8,10],['mma','B QK',10,12],['soft','B',12,14],['mma','B PV',14,16]];
    const overlap=[['load','A',0,2],['load','B',2,4],['mma','A QK',2,4],['mma','B QK',4,6],['soft','A',4,6],['soft','B',6,8],['mma','A PV',6,8],['mma','B PV',8,10]];
    const lanes=[['load','搬运 / TMA'],['mma','Tensor Core'],['soft','softmax']];
    const render=()=>{
      const tasks=pipeline.querySelector('select').value==='overlap'?overlap:serial;
      pipeline.querySelector('input').value=time;
      pipeline.querySelector('.fa-counter').textContent=`t = ${time}`;
      pipeline.querySelector('[data-action=prev]').disabled=time===0;
      pipeline.querySelector('[data-action=next]').disabled=time===16;
      pipeline.querySelector('[data-action=reset]').disabled=time===0;
      const name=task=>`${task[0]==='load'?'加载 ':task[0]==='soft'?'softmax ':''}${task[1]}`;
      pipeline.querySelector('.fa-scene').innerHTML=`<div class="fa-time-scroll" tabindex="0" role="region" aria-label="流水线时间轴，可横向滚动"><div class="fa-timeline"><div class="fa-axis-row"><span>阶段</span><div>${seq(17).map(t=>`<small style="left:${t/16*100}%">${t}</small>`).join('')}</div></div>${lanes.map(([key,label])=>`<div class="fa-lane"><strong>${label}</strong><div class="fa-track"><span class="fa-cursor" style="left:${time/16*100}%"></span>${tasks.filter(t=>t[0]===key).map(t=>`<span class="fa-job fa-${key} ${time>=t[3]?'fa-done':time>=t[2]?'fa-running':'fa-waiting'}" style="grid-column:${t[2]+1}/span ${t[3]-t[2]}" title="${name(t)}：阶段 ${t[2]} 到 ${t[3]}">${name(t)}</span>`).join('')}</div></div>`).join('')}</div></div><div class="fa-active-jobs">${lanes.map(([key,label])=>{const active=tasks.find(t=>t[0]===key&&t[2]<=time&&time<t[3]);return `<div><span>${label}</span><strong>${active?name(active):'此刻无任务'}</strong></div>`;}).join('')}</div><p class="fa-legend">粗边框＝执行中；✓＝已完成；竖线＝当前阶段。</p>`;
      const active=tasks.filter(t=>t[2]<=time&&time<t[3]).map(name);
      const doneA=time>=tasks.find(t=>t[1]==='A PV')[3],doneB=time>=tasks.find(t=>t[1]==='B PV')[3];
      pipeline.querySelector('.fa-status').textContent=doneA&&doneB?'A、B 均已完成。时间轴保留相同尺度，便于比较依赖安排；空白不是实测 GPU 空闲时间。':`t=${time}：${active.join(' 与 ')}。${active.length>1?'不同工作使用不同资源，可以重叠。':'同一 tile 的前置结果仍需先完成。'}`;
    };
    pipeline.querySelector('select').addEventListener('change',render);
    pipeline.querySelector('input').addEventListener('input',e=>{time=+e.target.value;render();});
    pipeline.addEventListener('click',e=>{const a=e.target.closest('[data-action]');if(!a)return;time=a.dataset.action==='reset'?0:Math.max(0,Math.min(16,time+(a.dataset.action==='next'?1:-1)));render();});render();
  }

  const tmem=mount('fa-tmem','矩阵中间结果放在哪里？',
    '切换硬件，沿 QK → softmax → PV 看数据位置。TMA 负责搬运，TMEM 是存储；它们不是同一个东西。',
    '<div class="fa-options"><label>代表性硬件路线<select data-option="hardware"><option value="hopper">H100 / Hopper</option><option value="blackwell">B200 / Blackwell SM100</option></select></label></div>',
    '图中 P 是当前块的指数权重，最终仍需按行归一化。Hopper 的 P 来源随 kernel 配置变化；B200 的 TMEM / tcgen05 路径不能直接套到 RTX 5090 / SM120。');
  if(tmem){
    const render=()=>{
      const b=tmem.querySelector('select').value==='blackwell';
      const nodes=b?[['QK 累加','S 在 TMEM','tcgen05.mma'],['线程处理 softmax','从 TMEM 读到寄存器','max / exp / sum'],['P 供给 PV','P 写回 TMEM，供 MMA 读取','PV 的输出累加在 TMEM']]:[['QK 累加','S 在线程寄存器','wgmma.mma_async'],['线程处理 softmax','在寄存器中计算','max / exp / sum'],['P 供给 PV','P 来自寄存器 / shared memory','PV 的输出累加在寄存器']];
      tmem.querySelector('.fa-scene').innerHTML=`<div class="fa-flow">${nodes.map((n,i)=>`${i?'<span class="fa-flow-arrow" aria-hidden="true">→</span>':''}<div class="fa-flow-node ${i===1?'fa-orange':i===0?'fa-blue':'fa-green'}"><strong>${n[0]}</strong><b>${n[1]}</b><span>${n[2]}</span></div>`).join('')}</div><div class="fa-equation">${b?'TMEM 保存数据；指数和归约仍需线程执行。减少中转，不等于取消数据移动。':'WGMMA 由 4 个 warp 组成的 warpgroup 协作。结果分布在线程寄存器中，不是每个线程各存一整块。'}</div>${b?'<div class="fa-two fa-bottlenecks"><div><strong>指数单元跟不上</strong><p>FA4 将部分指数计算分流到 FMA 多项式路径。</p></div><div><strong>旧输出反复缩放</strong><p>条件允许时保持旧参考值，减少重缩放。</p></div></div>':''}`;
      tmem.querySelector('.fa-status').textContent=b?'B200 的 MMA 更快后，指数吞吐与片上数据流更容易成为瓶颈，FA4 因而同时调整算法和流水线。':'H100 的 TMA 与异步 WGMMA 提供更多重叠机会；FA3 要把这些能力组织成有正确依赖的流水线。';
    };
    tmem.querySelector('select').addEventListener('change',render);render();
  }

  // Deep links into optional reference material must reveal their target first.
  function reveal(hash) {
    if(!hash)return;
    let target;
    try { target=document.getElementById(decodeURIComponent(hash.slice(1))); } catch { return; }
    if(!target)return;
    let details=target.closest('details'),opened=false;
    while(details){if(!details.open){details.open=true;opened=true;}details=details.parentElement?.closest('details');}
    if(opened)requestAnimationFrame(()=>target.scrollIntoView({block:'start'}));
  }
  document.addEventListener('click',event=>{
    const a=event.target.closest('a[href]');if(!a)return;
    const url=new URL(a.href,location.href);
    if(url.origin===location.origin&&url.pathname===location.pathname)reveal(url.hash);
  });
  window.addEventListener('hashchange',()=>reveal(location.hash));
  reveal(location.hash);
})();
