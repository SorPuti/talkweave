'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const chapters = CONTENT.chapters;
let cleanups = [], current = -1, currentFile = '';
let stored = {};
try { stored = JSON.parse(localStorage.getItem('talkweave-progress') || '{}'); } catch {}
const save = () => {try {localStorage.setItem('talkweave-progress',JSON.stringify(stored));} catch {}};
function toast(message) { $('#toast').textContent=message;$('#toast').classList.add('visible');setTimeout(()=>$('#toast').classList.remove('visible'),2300); }
async function copy(text) {try {await navigator.clipboard.writeText(text);toast('Código copiado.');} catch {toast('Selecione o código para copiar neste navegador.');}}
function nav() {
  let group='';
  $('#navigation').innerHTML = `<a class="nav-link ${current===-1?'active':''}" href="#inicio"><span class="number">⌂</span><span>Visão geral</span></a>` + chapters.map((c,i)=>{
    const label=c.group!==group?`<div class="nav-group">${esc(c.group)}</div>`:'';group=c.group;
    return label+`<a class="nav-link ${current===i?'active':''} ${stored[c.id]?'done':''}" href="#${c.id}" ${current===i?'aria-current="page"':''}><span class="number">${String(i+1).padStart(2,'0')}</span><span>${esc(c.title)}</span></a>`;
  }).join('');
  const count=chapters.filter(c=>stored[c.id]).length;
  $('#progress-label').textContent=`${count} / ${chapters.length}`;
  $('#progress-fill').style.width=`${count/chapters.length*100}%`;
}
function wave(className,phase=0) { return `<div class="wave ${className}" aria-hidden="true">${Array.from({length:90},(_,i)=>{const amplitude=(Math.sin(i*.43+phase)*.4+.6)*(Math.sin(i*.13+phase)*.4+.6);return `<b style="height:${3+amplitude*49}px;opacity:${.4+amplitude*.6}"></b>`;}).join('')}</div>`; }
function home() {
  return `<section class="hero"><div class="hero-copy"><div class="eyebrow">ENGENHARIA DE VOZ, PASSO A PASSO</div><h1>Da onda<br>à <em>conversa.</em></h1><p>Construa uma rede neural que escuta enquanto responde. Entenda cada camada, escreva o código e descubra o que funciona — começando em um computador comum.</p><div class="hero-actions"><a class="btn-primary" href="#mapa">Começar o tutorial <span>↗</span></a><button class="btn-secondary" data-files>Explorar o código <span>→</span></button></div><div style="margin-top:18px;font-size:11px"><a href="agents/" style="color:var(--teal)">◈ Passe um link ao seu agente de IA →</a></div><div class="hero-foot"><span>● PYTHON + PYTORCH</span><span>● PORTUGUÊS & INGLÊS</span><span>● CPU PRIMEIRO</span></div></div><div class="hero-card"><div class="card-caption"><span>DO SINAL À RESPOSTA</span><span class="live"><i></i> ARQUITETURA CAUSAL</span></div><div class="card-title">Dois fluxos. Uma conversa.</div><div class="wave-label"><span>01 / ENTRADA DO USUÁRIO</span><span>u[t]</span></div>${wave('',0)}<div class="wave-label"><span>02 / SAÍDA DO AGENTE</span><span>a[t]</span></div>${wave('agent',1.8)}<div class="track-note">A entrada continua sendo processada durante a saída.</div><div class="card-events"><span>FALA</span><span>RISADA</span><span>RUÍDO</span><span>TEXTO</span></div><div class="card-bottom"><span>DIAGRAMA ILUSTRATIVO</span><span>20 ms / quadro no laboratório</span></div></div></section>
  <div class="metrics-strip"><div class="metric"><strong>${chapters.length} capítulos</strong><small>Do primeiro array ao streaming</small></div><div class="metric"><strong>100% local</strong><small>Laboratório sem API paga</small></div><div class="metric"><strong>CPU primeiro</strong><small>Uma rede pequena, feita do zero</small></div><div class="metric"><strong>Código aberto</strong><small>Arquivos completos · licença MIT</small></div></div>
  <section><div class="section-label">UM CAMINHO, TRÊS NÍVEIS</div><div class="section-title"><h2>Construa entendendo.</h2><span>Aprenda → execute → meça → melhore</span></div><div class="path-grid"><a class="path-card" href="#sinal"><div class="path-num"><span>01 / FUNDAMENTOS</span><span>↗</span></div><h3>Aprenda a ouvir.</h3><p>Transforme áudio em dados. Explore sinais, gradientes, palavras, risadas e ruídos.</p><div class="path-tags"><span class="tag">NumPy</span><span class="tag">Dados</span><span class="tag">Classificação</span></div></a><a class="path-card" href="#codec"><div class="path-num"><span>02 / REDE NEURAL</span><span>↗</span></div><h3>Dê voz ao modelo.</h3><p>Programe codec, rede causal e treinamento. Gere áudio e texto, quadro a quadro.</p><div class="path-tags"><span class="tag">PyTorch</span><span class="tag">GRU</span><span class="tag">Full-duplex</span></div></a><a class="path-card" href="#avaliacao"><div class="path-num"><span>03 / QUALIDADE</span><span>↗</span></div><h3>Prove que funciona.</h3><p>Meça precisão, tempo e inteligibilidade. Encontre os erros antes de escalar.</p><div class="path-tags"><span class="tag">Avaliação</span><span class="tag">CPU</span><span class="tag">PersonaPlex</span></div></a></div></section>
  <div class="home-bottom"><section class="quick-card"><div class="panel-top"><span>Seu primeiro experimento</span><small>TERMINAL / CPU</small></div><pre><code><span style="color:#7f9d75"># Depois de instalar o ambiente do capítulo 02</span>\npython src/baseline.py\npython src/fixture.py\npython -m pytest -q</code></pre><div class="panel-foot">O primeiro conjunto usa tons sintéticos. Voz humana vem depois.</div></section><section class="manifest-card"><div class="section-label">SEM CAIXA-PRETA</div><h3>Executável não significa preciso.</h3><p>Você recebe código, testes e resultados reais — inclusive um treino que superajusta. O manual explica o que o modelo faz, o que ainda não aprendeu e como conferir.</p><a href="#receitas">Ver execução e resultados <span>↗</span></a></section></div>
  <section style="margin-top:42px"><div class="section-label">MAPA DA ARQUITETURA</div><div class="section-title"><h2>Ouvir e responder no mesmo relógio.</h2><span>Rede nativa, causal e pequena</span></div>${labHTML('architecture','home')}</section>`;
}
function blockHTML(b,index) {
  if(b.type==='p')return `<p>${esc(b.text)}</p>`;
  if(b.type==='h')return `<h2 id="sec-${index}">${esc(b.text)}</h2>`;
  if(b.type==='note')return `<div class="note ${b.kind==='warn'?'warn':''}"><span aria-hidden="true">${b.kind==='warn'?'△':'↳'}</span><div>${esc(b.text)}</div></div>`;
  if(b.type==='table')return `<div class="table-wrap"><table><thead><tr>${b.headers.map(x=>`<th scope="col">${esc(x)}</th>`).join('')}</tr></thead><tbody>${b.rows.map(r=>`<tr>${r.map(x=>`<td>${esc(x)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  if(b.type==='code')return `<div class="code-example ${b.result?'has-output':''}"><div class="code-top"><span>${esc(b.file||b.lang.toUpperCase()+' / EXEMPLO')}</span><div>${b.file?`<button data-file="${esc(b.file)}">Arquivo completo ↗</button> &nbsp; `:''}<button data-copy="${index}">Copiar</button></div></div><pre><code>${esc(b.text)}</code></pre>${b.result?`<div class="code-result"><strong>O QUE OBSERVAR</strong><span>${esc(b.result)}</span></div>`:''}</div>`;
  if(b.type==='lab')return labHTML(b.name,`ch-${index}`);
  return '';
}
function article(ch,index) {
  const headings=ch.blocks.map((b,i)=>b.type==='h'?`<a href="#sec-${i}" data-anchor="sec-${i}">${esc(b.text)}</a>`:'').join('');
  const prev=index?chapters[index-1]:null,next=chapters[index+1];
  return `<div class="chapter-intro"><div class="eyebrow">${esc(ch.group)} <span>/ ${String(index+1).padStart(2,'0')}</span></div><h1>${esc(ch.title)}</h1><div class="subtitle">${esc(ch.subtitle)}</div><div class="chapter-meta"><span>≈ ${ch.minutes} MIN · ESTUDO GUIADO</span><span>PYTHON · PRÁTICA GUIADA</span><button class="print-button" id="print-chapter">Imprimir capítulo ↗</button></div></div><div class="article-layout"><article class="article">${ch.blocks.map(blockHTML).join('')}<div class="source-links">${ch.sources.map(id=>{const s=CONTENT.sources.find(s=>s[0]===id);return `<a href="${esc(s[2])}" target="_blank" rel="noopener noreferrer">${esc(s[1])} ↗</a>`}).join('')}</div><section class="chapter-end"><h3>Antes de seguir</h3>${ch.checks.map((text,i)=>`<label class="check-row"><input type="checkbox" data-check="${ch.id}-${i}" ${stored[ch.id+'-'+i]?'checked':''}>${esc(text)}</label>`).join('')}<button id="complete-chapter" class="complete-button ${stored[ch.id]?'checked':''}">${stored[ch.id]?'✓ Capítulo concluído':'Marcar capítulo como concluído'}</button><div class="chapter-navigation"><a href="#${prev?prev.id:'inicio'}"><small>← ANTERIOR</small>${esc(prev?prev.title:'Visão geral')}</a>${next?`<a href="#${next.id}"><small>PRÓXIMO →</small>${esc(next.title)}</a>`:`<a href="downloads/talkweave.zip" download><small>LEVE O PROJETO →</small>Baixar código completo</a>`}</div></section></article><aside class="article-aside"><div class="aside-title">NESTE CAPÍTULO</div>${headings||'<a href="#main">Introdução e prática</a>'}<div class="aside-tip">Leia o código.<br>Execute em partes.<br>Meça antes de escalar.<br><br><a href="downloads/talkweave-tutorial.md" download>↓ Manual em Markdown</a></div></aside></div>`;
}
const labTitles={console:'CONSOLE / RENDERIZAÇÃO SOBRE DADOS REAIS',architecture:'ARQUITETURA / FLUXOS PARALELOS',signal:'LAB 01 / EXPLORE UM SINAL',logistic:'LAB 02 / TREINE UMA REGRESSÃO',wer:'LAB 03 / MEÇA ERROS DE TEXTO',loss:'RESULTADO REAL / TREINO EM CPU',timeline:'LAB 04 / CONCORRÊNCIA E INTERRUPÇÃO',budget:'LAB 05 / MEMÓRIA E DADOS',artifacts:'RESULTADO REAL / ÁUDIOS DO EXPERIMENTO',files:'CÓDIGO COMPLETO / ABRIR E INSPECIONAR',verification:'VERIFICAÇÃO / ESCOPO DO TESTE',sources:'BIBLIOTECA / FONTES PRIMÁRIAS'};
function labHTML(name,id) {
  let body='';
  if(name==='console')body=`<img src="downloads/console.svg" alt="Console Rich com energia, tempo de inferência, perdas de fila e escores sobre o fixture sintético" style="display:block;width:100%;border-radius:6px"><div class="lab-caption">Renderização real da UI com um quadro do fixture de tons e inferência local medida. Não é uma gravação de microfone nem prova de precisão de voz. Use os comandos abaixo para o console interativo.</div>`;
  if(name==='architecture')body=`<div class="graph"><div class="graph-col"><div class="graph-node"><strong>Áudio do usuário</strong><small>320 amostras / 20 ms</small><span class="graph-arrow">→</span></div><div class="graph-node"><strong>História do agente</strong><small>a[t−1] · texto[t−1]</small><span class="graph-arrow">→</span></div></div><div class="graph-col"><div class="graph-node"><strong>Codec VQ</strong><small>onda → código</small><span class="graph-arrow">→</span></div><div class="graph-node"><strong>Embeddings</strong><small>códigos → vetores</small><span class="graph-arrow">→</span></div></div><div class="graph-core"><strong>GRU causal</strong><span>h[t] = f(x[t], h[t−1])</span><div style="margin-top:14px;font-size:10px;color:#5a7050">Estado preservado<br>Sem quadros futuros</div></div><div class="graph-col"><div class="graph-node orange"><strong>Áudio do agente</strong><small>código → decoder → onda</small></div><div class="graph-node"><strong>Dois textos</strong><small>usuário + agente</small></div><div class="graph-node"><strong>Eventos</strong><small>fala · riso · ruído</small></div></div></div><div class="graph-legend">A saída prevista do agente volta como história no próximo tick. O usuário continua entrando durante a geração.</div>`;
  if(name==='signal')body=`<div class="lab-controls"><label>Frequência <span data-value="freq">220 Hz</span><input data-control="freq" type="range" min="80" max="1600" value="220"></label><label>Ruído <span data-value="noise">0.05</span><input data-control="noise" type="range" min="0" max=".5" step=".01" value=".05"></label><button data-action="tone">▷ Ouvir tom · 0,6 s</button></div><canvas data-canvas="signal" width="900" height="240" role="img" aria-label="Forma de onda calculada do sinal sintético"></canvas><div class="lab-output" data-output></div><div class="lab-caption">Sinal sintético calculado localmente. RMS/ZCR são medidas reais; não há detector de fala neste gráfico. O áudio toca só o tom, com volume reduzido.</div>`;
  if(name==='logistic')body=`<div class="lab-controls"><label>Taxa de aprendizado<select data-control="lr"><option value=".005">0,005 · lenta</option><option value=".05" selected>0,05 · referência</option><option value=".5">0,5 · alta</option></select></label><button data-action="train">Treinar 300 passos</button></div><canvas data-canvas="logistic" width="900" height="240" role="img" aria-label="Curva de perda durante treinamento da regressão logística"></canvas><div class="lab-output" data-output>Pronto para treinar. Dados: tons versus ruído sintético; 300 exemplos de treino, 100 de teste.</div><div class="lab-caption">Gradiente calculado em JavaScript, sem API. Dados gerados com seed fixa. Não é uma medição de reconhecimento de voz humana.</div>`;
  if(name==='wer')body=`<div class="field-row"><label>Texto de referência<input data-control="reference" value="abra o contato da Ana"></label><label>Texto reconhecido<input data-control="hypothesis" value="abra contato da Ana"></label></div><div class="lab-output" data-output></div><div class="lab-caption">Distância de edição por palavra, com minúsculas e Unicode NFC. WER pode passar de 100% quando há muitas inserções. Não avalia timbre ou qualidade da resposta.</div>`;
  if(name==='loss')body=`<canvas data-canvas="loss" width="900" height="250" role="img" aria-label="Perda de treino diminui e perda de validação aumenta no experimento sintético"></canvas><div class="lab-output">Treino duplex: 9,43 → 0,42 · Validação: 9,17 → 4,78<br>Geração livre: texto vazio · WER dos rótulos artificiais: 100%</div><div class="lab-caption">Curvas de uma execução real de 100 passos, seed 7, em CPU. Fixture de tons, não fala. Validação do treinador usa uma amostra sorteada por registro; em um conjunto maior, agregue todas as sessões para comparar checkpoints.</div>`;
  if(name==='timeline')body=`<div class="timeline-head"><span>Simulação de uma sessão · 6 segundos</span><button class="lab-button" data-action="simulate">▷ Reproduzir fluxo</button></div><div class="timeline-track"><span>USUÁRIO</span><div class="timeline-bar"><span class="timeline-segment" style="left:0;width:28%">pedido</span><span class="timeline-segment" style="left:52%;width:23%">interrupção</span><i class="timeline-cursor"></i></div></div><div class="timeline-track"><span>AGENTE</span><div class="timeline-bar"><span class="timeline-segment agent" style="left:24%;width:29%">resposta em curso</span><span class="timeline-segment agent" style="left:77%;width:22%">nova resposta</span><i class="timeline-cursor"></i></div></div><div class="timeline-track"><span>WORKER</span><div class="timeline-bar"><span class="timeline-segment tool" style="left:32%;width:39%">operação assíncrona</span><i class="timeline-cursor"></i></div></div><div class="lab-output" data-output>Captura sempre ativa. O gráfico é uma simulação de agendamento, não um modelo executando.</div><div class="lab-caption">Na aplicação real, a detecção de interrupção precisa distinguir fala, eco e ruído. O worker ilustrado não está conectado ao modelo pequeno do ZIP.</div>`;
  if(name==='budget')body=`<div class="lab-controls"><label>Parâmetros <span data-value="params">1 milhão</span><input data-control="params" type="range" min="1" max="100" value="1"></label><label>Áudio estéreo <span data-value="hours">1 hora</span><input data-control="hours" type="range" min="1" max="100" value="1"></label></div><div class="lab-output" data-output></div><div class="lab-caption">Estimativa aritmética: pesos FP32=4 B/parâmetro; pesos+gradientes+Adam ≈16 B/parâmetro, antes de ativações. WAV PCM16 a 16 kHz, 2 canais. Não é medição de RAM/VRAM nem orçamento de computador.</div>`;
  if(name==='artifacts')body=`<div class="audio-grid"><div class="audio-item">Entrada sintética<small>Senoide do conjunto de validação. Sem voz humana.</small><audio controls preload="none" src="downloads/validation-user.wav"></audio></div><div class="audio-item">Saída livre do modelo<small>Modelo treinado por 100 passos com tons. Não é fala.</small><audio controls preload="none" src="downloads/reply.wav"></audio></div></div><div class="lab-caption">Volume inicial baixo recomendado para qualquer áudio gerado. <a href="downloads/experiment-results.json" download style="text-decoration:underline">Baixar métricas do experimento ↗</a></div>`;
  if(name==='files')body=`<div class="file-buttons">${Object.keys(CONTENT.files).filter(f=>/\.py$|README/.test(f)).map(f=>`<button data-file="${esc(f)}">${esc(f)} ↗</button>`).join('')}</div><div class="lab-caption">Todos os arquivos acima podem ser lidos aqui e estão no ZIP. <a href="downloads/talkweave-tutorial.md" download style="text-decoration:underline">Baixar o manual completo em Markdown.</a></div>`;
  if(name==='verification')body=`<div class="verification">${esc(CONTENT.verification)}</div><div class="lab-caption">Inferência em arquivo, não microfone. Sem GPU disponível neste ambiente. Nenhuma métrica de voz humana é afirmada.</div>`;
  if(name==='sources')body=`<div class="reference-list">${CONTENT.sources.map((s,i)=>`<a class="reference" href="${esc(s[2])}" target="_blank" rel="noopener noreferrer"><span class="ref-num">${String(i+1).padStart(2,'0')}</span><div><strong>${esc(s[1])} ↗</strong><p>${esc(s[3])}</p></div></a>`).join('')}</div>`;
  return `<section class="lab" data-lab="${name}" id="lab-${id}"><div class="lab-bar"><span>${labTitles[name]}</span><span>${['signal','logistic','wer','budget'].includes(name)?'INTERATIVO':name==='timeline'?'SIMULAÇÃO':'DOCUMENTADO'}</span></div><div class="lab-body">${body}</div></section>`;
}
function drawChart(canvas,series,{max=null,labels=[]}={}) {
  const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#f8faf5';ctx.fillRect(0,0,w,h);
  const pad={l:45,r:24,t:28,b:35};const ww=w-pad.l-pad.r,hh=h-pad.t-pad.b;
  const upper=max||Math.max(...series.flatMap(s=>s.values))*1.08||1;
  ctx.font='11px monospace';ctx.fillStyle='#7d8c74';ctx.strokeStyle='#dfe7d8';ctx.lineWidth=1;
  for(let i=0;i<5;i++){let y=pad.t+hh*i/4;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillText((upper*(1-i/4)).toFixed(1),5,y+4);}
  series.forEach((s,idx)=>{ctx.strokeStyle=s.color;ctx.lineWidth=2.5;ctx.beginPath();s.values.forEach((v,i)=>{let x=pad.l+ww*i/Math.max(1,s.values.length-1),y=pad.t+hh*(1-v/upper);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();ctx.fillStyle=s.color;ctx.fillText(s.name,pad.l+idx*200,15);});
  if(labels.length){ctx.fillStyle='#7d8c74';labels.forEach((s,i)=>ctx.fillText(s,pad.l+ww*i/Math.max(1,labels.length-1)-10,h-10));}
}
function initLabs() {
  $$('[data-lab]').forEach(el=>{
    const name=el.dataset.lab;const q=s=>el.querySelector(s);
    if(name==='signal'){
      const update=()=>{const f=+q('[data-control="freq"]').value,n=+q('[data-control="noise"]').value;
        q('[data-value="freq"]').textContent=f+' Hz';q('[data-value="noise"]').textContent=n.toFixed(2);
        let samples=Array.from({length:320},(_,i)=>.3*Math.sin(2*Math.PI*f*i/16000)+n*(Math.sin(i*73.71)*.6+Math.sin(i*31.43)*.4));
        let rms=Math.sqrt(samples.reduce((a,b)=>a+b*b,0)/320);let crosses=samples.slice(1).filter((x,i)=>(x<0)!==(samples[i]<0)).length/319;
        const c=q('canvas'),ctx=c.getContext('2d');ctx.fillStyle='#f8faf5';ctx.fillRect(0,0,c.width,c.height);ctx.strokeStyle='#dce5d4';for(let i=0;i<5;i++){ctx.beginPath();ctx.moveTo(0,i*60);ctx.lineTo(c.width,i*60);ctx.stroke();}
        ctx.strokeStyle='#348563';ctx.lineWidth=2;ctx.beginPath();samples.forEach((v,i)=>{const x=i*c.width/319,y=120-v*125;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
        q('[data-output]').textContent=`320 amostras · 20 ms · RMS ${rms.toFixed(3)} · ${(20*Math.log10(rms)).toFixed(1)} dBFS · ZCR ${crosses.toFixed(3)}`;
      };
      el.querySelectorAll('input').forEach(i=>i.addEventListener('input',update));update();
      q('[data-action="tone"]').onclick=async()=>{try{const Audio=window.AudioContext||window.webkitAudioContext;const ac=new Audio();await ac.resume();const o=ac.createOscillator(),g=ac.createGain();o.frequency.value=+q('[data-control="freq"]').value;g.gain.setValueAtTime(0,ac.currentTime);g.gain.linearRampToValueAtTime(.07,ac.currentTime+.02);g.gain.setValueAtTime(.07,ac.currentTime+.55);g.gain.linearRampToValueAtTime(0,ac.currentTime+.6);o.connect(g).connect(ac.destination);o.start();o.stop(ac.currentTime+.62);o.onended=()=>ac.close();cleanups.push(()=>{if(ac.state!=='closed')ac.close();});}catch{toast('Áudio indisponível neste navegador.');}};
    }
    if(name==='logistic'){
      drawChart(q('canvas'),[{name:'loss de treino',color:'#2a8160',values:[.69,.69]}],{max:.8});
      q('[data-action="train"]').onclick=()=>{
        let state=7;const random=()=>{state=(1664525*state+1013904223)>>>0;return (state+.5)/4294967296;};
        const normal=()=>Math.sqrt(-2*Math.log(random()))*Math.cos(2*Math.PI*random());
        const data=Array.from({length:400},(_,i)=>{const y=i%2;return {x:[.09+normal()*.012,y?.46+normal()*.08:.04+normal()*.02],y};});
        for(let i=data.length-1;i>0;i--){const j=Math.floor(random()*(i+1));[data[i],data[j]]=[data[j],data[i]];}
        const tr=data.slice(0,300),te=data.slice(300);const mean=[0,1].map(j=>tr.reduce((a,r)=>a+r.x[j],0)/300),sd=[0,1].map(j=>Math.sqrt(tr.reduce((a,r)=>a+(r.x[j]-mean[j])**2,0)/300)+1e-8);
        const normalize=r=>({...r,z:r.x.map((x,j)=>(x-mean[j])/sd[j])});const train=tr.map(normalize),test=te.map(normalize);
        let w=[0,0],b=0;const lr=+q('[data-control="lr"]').value,losses=[];const sigmoid=z=>1/(1+Math.exp(-Math.max(-30,Math.min(30,z))));
        for(let step=0;step<300;step++){let dw=[0,0],db=0,loss=0;for(const r of train){const p=sigmoid(w[0]*r.z[0]+w[1]*r.z[1]+b),e=p-r.y;dw[0]+=e*r.z[0];dw[1]+=e*r.z[1];db+=e;loss-=r.y*Math.log(p+1e-9)+(1-r.y)*Math.log(1-p+1e-9);}w=w.map((v,j)=>v-lr*(dw[j]/300+.01*v));b-=lr*db/300;if(step%5===0)losses.push(loss/300);}
        const acc=test.filter(r=>(sigmoid(w[0]*r.z[0]+w[1]*r.z[1]+b)>=.5)==Boolean(r.y)).length/100;
        drawChart(q('canvas'),[{name:'entropia cruzada / treino',color:'#2a8160',values:losses}],{max:.8,labels:['0','150','300 passos']});
        q('[data-output]').textContent=`Treinado: 300 passos · loss ${losses.at(-1).toFixed(4)} · teste sintético ${(acc*100).toFixed(0)}%\nw = [${w.map(v=>v.toFixed(3)).join(', ')}] · b = ${b.toFixed(3)}\nÉ uma tarefa artificial fácil. Esse percentual não mede voz humana.`;
      };
    }
    if(name==='wer'){
      const update=()=>{const tokenize=s=>s.normalize('NFC').toLowerCase().trim().split(/\s+/).filter(Boolean),a=tokenize(q('[data-control="reference"]').value),b=tokenize(q('[data-control="hypothesis"]').value);let row=Array.from({length:b.length+1},(_,i)=>i);a.forEach((x,i)=>{const next=[i+1];b.forEach((y,j)=>next.push(Math.min(next[j]+1,row[j+1]+1,row[j]+(x!==y))));row=next;});const edits=row[b.length];q('[data-output]').textContent=`${edits} edição(ões) / ${a.length} palavra(s) de referência → WER ${(100*edits/Math.max(1,a.length)).toFixed(1)}%${a.length?'':' · referência vazia: denominador protegido em 1'}`;};
      el.querySelectorAll('input').forEach(i=>i.addEventListener('input',update));update();
    }
    if(name==='loss')drawChart(q('canvas'),[{name:'treino / teacher forcing',color:'#268365',values:CONTENT.metrics.map(r=>r.loss)},{name:'validação / teacher forcing',color:'#c17d45',values:CONTENT.metrics.map(r=>r.validation_loss)}],{labels:CONTENT.metrics.map(r=>String(r.step))});
    if(name==='timeline'){
      let timer=null;const button=q('[data-action="simulate"]');
      button.onclick=()=>{if(timer)clearInterval(timer);button.disabled=true;let start=performance.now();timer=setInterval(()=>{const t=Math.min(6,(performance.now()-start)/1000);el.querySelectorAll('.timeline-cursor').forEach(c=>c.style.left=t/6*100+'%');q('[data-output]').textContent=t<1.45?'Usuário fala. A rede atualiza o estado a cada quadro.':t<3.12?'Agente responde. Captura e worker continuam em paralelo.':t<4.5?'Usuário interrompe. Simulação descarta a saída antiga; entrada permanece ativa.':'Nova resposta. Resultado do worker exige validação de turno e estado.';if(t>=6){clearInterval(timer);timer=null;button.disabled=false;button.textContent='↻ Repetir simulação';}},40);};cleanups.push(()=>{if(timer)clearInterval(timer);});
    }
    if(name==='budget'){
      const update=()=>{const p=+q('[data-control="params"]').value,h=+q('[data-control="hours"]').value;q('[data-value="params"]').textContent=p+' milhão(ões)';q('[data-value="hours"]').textContent=h+' hora(s)';q('[data-output]').textContent=`Pesos FP32: ${(p*1e6*4/1024**2).toFixed(1)} MiB\nPesos + gradientes + estados Adam: ≈ ${(p*1e6*16/1024**2).toFixed(1)} MiB + ativações\nÁudio WAV PCM16 estéreo: ${(h*3600*16000*2*2/1024**3).toFixed(2)} GiB\nÁudio float32 carregado: ${(h*3600*16000*2*4/1024**3).toFixed(2)} GiB + cópias e anotações`;};el.querySelectorAll('input').forEach(i=>i.addEventListener('input',update));update();
    }
  });
}
function showFile(name) {
  if(!CONTENT.files[name])return;
  currentFile=name;$('#file-name').textContent=name;$('#file-source').textContent=CONTENT.files[name];
  $('#file-list').innerHTML=Object.keys(CONTENT.files).map(f=>`<button class="${f===name?'active':''}" data-select-file="${esc(f)}">${esc(f)}</button>`).join('');
  $('#file-list').querySelectorAll('button').forEach(b=>b.onclick=()=>showFile(b.dataset.selectFile));
  if(!$('#file-dialog').open)$('#file-dialog').showModal();
}
function render() {
  const hash=decodeURIComponent(location.hash.slice(1)||'inicio');
  if(hash.startsWith('sec-'))return;
  cleanups.forEach(f=>f());cleanups=[];
  current=chapters.findIndex(c=>c.id===hash);
  $('#main').innerHTML=current<0?home():article(chapters[current],current);
  $('#crumb').textContent=current<0?'Visão geral':chapters[current].title;
  document.title=(current<0?'Da onda à conversa':chapters[current].title)+' — TalkWeave';
  nav();initLabs();window.scrollTo({top:0,behavior:'instant'});
  $('#sidebar').classList.remove('open');$('#menu-toggle').setAttribute('aria-expanded','false');
  $$('[data-copy]').forEach(b=>b.onclick=()=>copy(chapters[current].blocks[+b.dataset.copy].text));
  $$('[data-file]').forEach(b=>b.onclick=()=>showFile(b.dataset.file));
  $$('[data-files]').forEach(b=>b.onclick=()=>showFile('README.md'));
  $$('[data-anchor]').forEach(a=>a.onclick=e=>{e.preventDefault();document.getElementById(a.dataset.anchor)?.scrollIntoView({behavior:'smooth'});});
  $$('[data-check]').forEach(c=>c.onchange=()=>{stored[c.dataset.check]=c.checked;save();});
  if($('#complete-chapter'))$('#complete-chapter').onclick=()=>{const id=chapters[current].id;stored[id]=!stored[id];save();nav();const b=$('#complete-chapter');b.classList.toggle('checked',!!stored[id]);b.textContent=stored[id]?'✓ Capítulo concluído':'Marcar capítulo como concluído';toast(stored[id]?'Progresso salvo neste navegador.':'Capítulo marcado como pendente.');};
  if($('#print-chapter'))$('#print-chapter').onclick=()=>window.print();
}
function search() {
  const query=$('#search-input').value.normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase();
  const results=chapters.filter(c=>`${c.title} ${c.subtitle} ${c.blocks.map(b=>b.text||'').join(' ')}`.normalize('NFD').replace(/[\u0300-\u036f]/g,'').toLowerCase().includes(query));
  $('#search-results').innerHTML=results.length?results.map(c=>`<a class="search-result" href="#${c.id}">${esc(c.title)}<small>${esc(c.subtitle)}</small></a>`).join(''):'<div class="search-empty">Nenhum capítulo encontrado. Tente “áudio”, “treino” ou “eventos”.</div>';
  $('#search-results').querySelectorAll('a').forEach(a=>a.onclick=()=>$('#search-dialog').close());
}
$('#search-open').onclick=()=>{$('#search-dialog').showModal();search();$('#search-input').focus();};
$('#search-close').onclick=()=>$('#search-dialog').close();$('#search-input').oninput=search;
$('#files-open').onclick=()=>showFile('README.md');$('#file-close').onclick=()=>$('#file-dialog').close();$('#file-copy').onclick=()=>copy(CONTENT.files[currentFile]);
$('#menu-toggle').onclick=()=>{const open=$('#sidebar').classList.toggle('open');$('#menu-toggle').setAttribute('aria-expanded',String(open));};
document.addEventListener('keydown',e=>{if(e.key==='/'&&!['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)&&!$('#file-dialog').open){e.preventDefault();$('#search-open').click();}});
document.addEventListener('click',e=>{if(window.innerWidth<=720&&!e.target.closest('#sidebar')&&!e.target.closest('#menu-toggle')){$('#sidebar').classList.remove('open');$('#menu-toggle').setAttribute('aria-expanded','false');}});
window.addEventListener('hashchange',render);render();
