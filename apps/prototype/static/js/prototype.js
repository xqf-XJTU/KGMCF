const state = {
  features: [],
  methods: [],
  workflow: [],
  currentFeature: '',
  detectedFeature: '',
  currentMethod: 'KGMCF',
  currentWorkflow: null,
  history: [],
  datasetPage: 1,
  uploadedImage: '',
  reasonUploadedImage: '',
  graph: null,
  graphView: { scale: 1, x: 0, y: 0 },
  promptTemplates: {},
  promptTab: 'system',
  runtime: {},
  trainingTraceLoaded: false
};

const $ = (id) => document.getElementById(id);
const on = (id, event, fn) => { const el = $(id); if (el) el.addEventListener(event, fn); };
const esc = (v) => String(v ?? '').replace(/[&<>'"]/g, s => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[s]));
const imgUrl = (path) => path ? `/api/image?path=${encodeURIComponent(path)}` : '';
async function api(url, options = {}) {
  const r = await fetch(url, options);
  if (!r.ok) {
    const m = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${m.slice(0, 220)}`);
  }
  return await r.json();
}
function fillSelect(el, items, labelFn, valueFn) {
  if (!el) return;
  el.innerHTML = items.map(x => `<option value="${esc(valueFn(x))}">${esc(labelFn(x))}</option>`).join('');
}
function featureLabel(f) { return `${f.feature_id} · ${f.feature_name || f.feature_id}`; }
function kvs(items) { return items.map(([k, v]) => `<div class="kv"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join(''); }
function currentFeatureLabel() {
  const f = state.features.find(x => x.feature_id === state.currentFeature);
  return f ? `${f.feature_id} · ${f.feature_name}` : (state.currentFeature || 'pending');
}
function setSelects() {
  const dataset = $('datasetFeatureSelect');
  if (dataset) {
    fillSelect(dataset, [{ feature_id: 'all', feature_name: 'All categories' }, ...state.features], featureLabel, x => x.feature_id);
    dataset.value = dataset.value || 'all';
  }
  const proto = $('prototypeFeatureSelect');
  if (proto) {
    fillSelect(proto, state.features, featureLabel, x => x.feature_id);
    if (state.currentFeature && [...proto.options].some(o => o.value === state.currentFeature)) proto.value = state.currentFeature;
  }
  const method = $('reasonMethodSelect');
  if (method) { fillSelect(method, state.methods, x => x, x => x); method.value = state.currentMethod; }
  updateCurrentFeatureBadges();
}
function updateCurrentFeatureBadges() {
  const label = currentFeatureLabel();
  if ($('currentFeatureBadge')) $('currentFeatureBadge').textContent = label;
  if ($('physicsFeatureBadge')) $('physicsFeatureBadge').textContent = state.currentFeature ? `Current anchored feature: ${label}` : 'Current anchored feature: pending';
  if ($('reportSourceBadge')) {
    if (state.currentWorkflow?.feature) $('reportSourceBadge').textContent = `Generated from latest workflow: ${state.currentWorkflow.feature.feature_id} · ${state.currentWorkflow.feature.feature_name || ''}`;
    else $('reportSourceBadge').textContent = 'No process card generated yet. Run Inference first.';
  }
}
function metricCards(el, cards) {
  if (!el) return;
  el.innerHTML = cards.map(c => `<div class="metric"><span>${esc(c.label)}</span><strong>${esc(c.value)}</strong></div>`).join('');
}
function activatePage(page) {
  document.querySelectorAll('.nav-item[data-page]').forEach(b => b.classList.toggle('active', b.dataset.page === page));
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === `page-${page}`));
  if (page === 'dataset') loadDataset();
  if (page === 'prototypes') loadPrototypes();
  if (page === 'visual') loadVisual();
  if (page === 'physics') loadPhysics();
  if (page === 'engine') { loadEngine(false); loadRuntime(); }
  if (page === 'prompts') loadPrompts();
  if (page === 'reasoning') { updateReasonDrawing(); renderRuntimeSummary(); }
  if (page === 'report') loadReport();
  if (page === 'history') renderHistory();
  if (page === 'neo4j') renderBackendStatus('neo4jStatus');
  if (page === 'qwen') renderBackendStatus('qwenStatus');
}
async function init() {
  const data = await api('/api/app_state');
  state.features = data.features || [];
  state.methods = data.methods || [];
  state.workflow = data.workflow?.stages || [];
  state.runtime = data.model_runtime || {};
  if ($('healthBadge')) $('healthBadge').textContent = data.health?.status === 'ready' ? 'Service Ready' : (data.health?.status || 'Ready');
  setSelects();
  bindEvents();
  metricCards($('dashboardMetrics'), [
    { label: 'Aero-Instruct-5K', value: data.health?.sample_count || 0 },
    { label: 'CV20 categories', value: data.health?.feature_count || 0 },
    { label: 'Cognitive modules', value: 5 },
    { label: 'Planning methods', value: data.health?.method_count || 0 }
  ]);
  if ($('workflowSteps')) $('workflowSteps').innerHTML = state.workflow.map((s, i) => `<div class="stage"><div class="no">${i + 1}</div><div><b>${esc(s.title)}</b><small>${esc(s.description)}</small></div></div>`).join('');
  const selected = ['F00', 'F05', 'F11', 'F12'].map(fid => state.features.find(f => f.feature_id === fid)).filter(Boolean);
  if ($('featureTiles')) $('featureTiles').innerHTML = selected.map(f => `<div class="feature-tile" data-fid="${esc(f.feature_id)}"><b>${esc(featureLabel(f))}</b><div>${(f.risk_tags || []).map(t => `<span class="tag">${esc(t)}</span>`).join('')}</div></div>`).join('');
  document.querySelectorAll('.feature-tile').forEach(el => el.onclick = () => { state.currentFeature = el.dataset.fid; state.detectedFeature = state.currentFeature; setSelects(); activatePage('reasoning'); updateReasonDrawing(); });
  await loadDataset();
  await loadEngine(false);
  await loadPrompts();
  updateReasonDrawing();
  loadRuntime();
}
function bindEvents() {
  document.querySelectorAll('.nav-item[data-page]').forEach(b => b.onclick = () => activatePage(b.dataset.page));
  document.querySelectorAll('.nav-item.parent[data-group]').forEach(btn => {
    btn.onclick = () => {
      const group = $(btn.dataset.group);
      const expanded = group && group.classList.toggle('expanded');
      btn.classList.toggle('collapsed', !expanded);
      btn.classList.toggle('expanded', !!expanded);
    };
  });
  on('datasetSearchBtn', 'click', () => { state.datasetPage = 1; loadDataset(); });
  on('datasetSearch', 'keydown', e => { if (e.key === 'Enter') { state.datasetPage = 1; loadDataset(); } });
  on('datasetFeatureSelect', 'change', e => { state.datasetPage = 1; if (e.target.value !== 'all') state.currentFeature = e.target.value; loadDataset(); });
  on('prevPage', 'click', () => { state.datasetPage = Math.max(1, state.datasetPage - 1); loadDataset(); });
  on('nextPage', 'click', () => { state.datasetPage += 1; loadDataset(); });
  on('datasetExportBtn', 'click', () => { window.location = '/api/dataset/export'; });
  on('datasetImportBtn', 'click', () => alert('Seed records are managed through data/processed/aero_instruct_5k.jsonl and dataset_split_index.csv.'));
  on('prototypeFeatureSelect', 'change', e => { state.currentFeature = e.target.value; loadPrototypes(); updateCurrentFeatureBadges(); });
  on('refreshVisualBtn', 'click', loadVisual);
  on('visualOverviewBtn', 'click', loadVisual);
  on('fitGraphBtn', 'click', fitGraph);
  on('zoomInBtn', 'click', () => zoomGraph(1.18));
  on('zoomOutBtn', 'click', () => zoomGraph(0.85));
  on('visualAnchorBtn', 'click', runVisualAnchor);
  on('text-submit', 'click', e => { e.preventDefault(); submitNeo4jQuery('text', $('text-query')?.value || ''); });
  on('image-upload', 'change', e => { const f = e.target.files && e.target.files[0]; if (f) submitNeo4jQuery('image', f.name); uploadDrawingFromInput('image-upload'); });
  on('connectNeo4jBtn', 'click', connectNeo4jFromVisual);
  on('connectNeo4jMonitorBtn', 'click', connectNeo4jFromMonitor);
  on('reloadNeo4jStatusBtn', 'click', loadNeo4jStatus);
  on('syncPhysicsBtn', 'click', loadPhysics);
  on('reasonMethodSelect', 'change', e => { state.currentMethod = e.target.value; });
  on('reasonUpload', 'change', uploadReasonDrawing);
  on('anchorReasonBtn', 'click', anchorReasoningInput);
  on('executeReasoningBtn', 'click', executeReasoning);
  on('runQwenBtn', 'click', runQwenInference);
  on('downloadReportBtn', 'click', downloadCurrentWorkflow);
  on('downloadPdfBtn', 'click', downloadProcessPdf);
  on('saveRuntimeBtn', 'click', saveRuntime);
  on('testRuntimeBtn', 'click', testRuntime);
  on('saveEngineConfigBtn', 'click', saveEngineConfig);
  on('buildTrainingJobBtn', 'click', buildTrainingJob);
  on('loadTraceBtn', 'click', loadTrainingTrace);
  document.querySelectorAll('.prompt-tab').forEach(b => b.onclick = () => switchPromptTab(b.dataset.tab));
  on('savePromptBtn', 'click', savePrompt);
  on('testFusionBtn', 'click', testContextFusion);
}

async function loadDataset() {
  if (!$('datasetBody')) return;
  const fid = $('datasetFeatureSelect')?.value || 'all';
  const q = $('datasetSearch')?.value || '';
  const data = await api(`/api/dataset/samples?feature_id=${encodeURIComponent(fid)}&search=${encodeURIComponent(q)}&page=${state.datasetPage}&page_size=8`);
  if (state.datasetPage > data.page_count) { state.datasetPage = data.page_count || 1; return loadDataset(); }
  $('datasetBody').innerHTML = (data.records || []).map(r => `<tr><td><input type="checkbox"></td><td><b>${esc(r.sample_id)}</b><br><small>${esc(r.case_id)}</small></td><td>${r.visual_input ? `<img class="thumb" src="${imgUrl(r.visual_input)}" alt="view">` : ''}</td><td><span class="tag">${esc(r.feature_id)}</span><br>${esc(r.feature_name)}</td><td class="cot"><b>${esc(r.instruction)}</b><span>${esc(r.response_preview)}</span></td><td>${esc(r.split_id || '-')}<br><small>${esc(r.source_instance_id || '')}</small></td><td class="row-actions"><button onclick="openSampleFeature('${esc(r.feature_id)}')">Open</button><button onclick="openSampleFeature('${esc(r.feature_id)}','reasoning')">Run</button></td></tr>`).join('') || '<tr><td colspan="7" class="empty">No matching record</td></tr>';
  $('datasetTotal').textContent = `Total ${data.total || 0} Verified Instruction Pairs`;
  $('pageInfo').textContent = `${data.page || 1} / ${data.page_count || 1}`;
}
window.openSampleFeature = async (fid, page = 'visual') => {
  state.currentFeature = fid;
  state.detectedFeature = fid;
  const d = await api(`/api/features/${fid}`);
  if (page === 'reasoning') state.reasonUploadedImage = (d.image_examples || [])[0] || '';
  setSelects();
  activatePage(page);
  updateReasonDrawing();
};
async function loadPrototypes() {
  if (!$('prototypeGallery')) return;
  const fid = $('prototypeFeatureSelect')?.value || state.currentFeature || 'F00';
  const data = await api(`/api/prototypes?feature_id=${encodeURIComponent(fid)}&limit=24`);
  $('prototypeGallery').innerHTML = (data.records || []).map(r => `<div class="proto-card"><img src="${imgUrl(r.image_file)}" alt="${esc(r.feature_id)}"><b>${esc(r.feature_id)} · ${esc(r.feature_name)}</b><div>${(r.risk_tags || []).map(t => `<span class="tag">${esc(t)}</span>`).join('')}</div><small>${esc(r.sample_count)} instruction pairs</small></div>`).join('') || '<div class="empty">No visual prototype found.</div>';
}

async function connectNeo4jFromVisual() {
  const payload = { uri: $('neo4jUri')?.value || 'bolt://localhost:7687', username: $('neo4jUser')?.value || '', password: $('neo4jPassword')?.value || '' };
  const res = await api('/api/neo4j/connect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  renderNeo4jConnectStatus(res);
  await loadVisual();
}
async function connectNeo4jFromMonitor() {
  const payload = { uri: $('neo4jUriMonitor')?.value || 'bolt://localhost:7687', username: $('neo4jUserMonitor')?.value || '', password: $('neo4jPasswordMonitor')?.value || '' };
  const res = await api('/api/neo4j/connect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  renderNeo4jConnectStatus(res);
  await loadNeo4jStatus();
  if (document.querySelector('#page-visual.active')) await loadVisual();
}
function renderNeo4jConnectStatus(status) {
  const text = (status.connected ? 'Connected: ' : 'Not connected: ') + (status.message || '');
  ['neo4jConnectStatus'].forEach(id => { if ($(id)) { $(id).textContent = text; $(id).className = status.connected ? 'ok-text' : 'warn-text'; } });
  if ($('neo4jUri') && status.uri) $('neo4jUri').value = status.uri;
  if ($('neo4jUriMonitor') && status.uri) $('neo4jUriMonitor').value = status.uri;
  if ($('neo4jUser') && status.username) $('neo4jUser').value = status.username;
  if ($('neo4jUserMonitor') && status.username) $('neo4jUserMonitor').value = status.username;
}
async function loadNeo4jStatus() {
  const s = await api('/api/neo4j/status');
  renderNeo4jConnectStatus(s);
  if ($('neo4jStatus')) $('neo4jStatus').innerHTML = kvs([
    ['Connection', s.connected ? 'connected' : 'not connected'], ['URI', s.uri || '-'], ['Username', s.username || '-'], ['Password saved', 'false'], ['Nodes', s.node_count || 0], ['Relationships', s.relationship_count || 0], ['Message', s.message || '']
  ]);
  return s;
}
async function loadVisual() {
  const container = $('mynetwork');
  if (!container) return;
  const status = await api('/api/neo4j/status');
  renderNeo4jConnectStatus(status);
  if (!status.connected) {
    destroyVisNetwork();
    container.innerHTML = `<div class="graph-empty-box"><b>Neo4j Aero-MMKG not connected</b><br>${esc(status.message || 'Please enter Bolt URI, username, and password, then click Connect Neo4j Graph.')}</div>`;
    if ($('answer-box')) $('answer-box').textContent = '请先输入正确的 Neo4j 用户名和密码并连接知识图谱。';
    if ($('visualReasoning')) $('visualReasoning').innerHTML = kvs([
      ['Graph backend', 'Neo4j'], ['Connection', 'not connected'], ['Required input', 'Bolt URI, username, and password']
    ]);
    if ($('nodeInspector')) $('nodeInspector').innerHTML = '<div class="hint">No local or hard-coded graph is displayed. Connect Neo4j to load the actual Aero-MMKG.</div>';
    return;
  }
  if ($('answer-box')) $('answer-box').textContent = '图谱加载中...';
  const graph = await api('/api/graph_data');
  drawNeo4jVisNetwork(graph);
  if ($('visualReasoning')) $('visualReasoning').innerHTML = kvs([
    ['Graph backend', 'Neo4j connected'],
    ['Displayed graph', `${graph.nodes?.length || 0} nodes / ${graph.edges?.length || 0} relations`],
    ['Database graph', `${graph.status?.node_count || graph.nodes?.length || 0} nodes / ${graph.status?.relationship_count || graph.edges?.length || 0} relations`],
    ['Interaction', 'Drag nodes; wheel zoom; drag background; click node for details']
  ]);
  if ($('nodeInspector')) $('nodeInspector').innerHTML = '<div class="hint">Click a Neo4j node to inspect properties. Text and image queries can highlight related paths.</div>';
  if ($('answer-box')) $('answer-box').textContent = '图谱加载完成！您可以开始提问或点击节点查看属性。';
}
function destroyVisNetwork() {
  if (state.visNetwork) { state.visNetwork.destroy(); state.visNetwork = null; }
  state.visNodes = null; state.visEdges = null;
}
function neo4jColorMap() {
  return {
    '零件': { background: '#FFC107', border: '#FF9800' },
    'Part': { background: '#FFC107', border: '#FF9800' },
    'Feature': { background: '#FFC107', border: '#FF9800' },
    '工序': { background: '#03A9F4', border: '#0288D1' },
    'Operation': { background: '#03A9F4', border: '#0288D1' },
    '工步': { background: '#8BC34A', border: '#689F38' },
    'Step': { background: '#8BC34A', border: '#689F38' },
    'Strategy': { background: '#8BC34A', border: '#689F38' },
    '设备': { background: '#9C27B0', border: '#7B1FA2' },
    'Resource': { background: '#9C27B0', border: '#7B1FA2' },
    'Equipment': { background: '#9C27B0', border: '#7B1FA2' },
    'Tool': { background: '#E91E63', border: '#C2185B' },
    '刀具': { background: '#E91E63', border: '#C2185B' },
    '分析': { background: '#607D8B', border: '#455A64' },
    'Physics Model': { background: '#607D8B', border: '#455A64' },
    'Image': { background: '#FFFFFF', border: '#BDBDBD' },
    'Constraint': { background: '#38a7e4', border: '#0288D1' }
  };
}
function normalizeVisNode(n) {
  const group = n.group || n.neo4j_label || 'Node';
  const label = String(n.label || n.name || n.id);
  const node = {
    id: n.id,
    label,
    group,
    title: typeof n.title === 'string' ? n.title : JSON.stringify(n.title || {}, null, 2),
    font: n.font || { color: '#222', size: 14 },
    size: n.size || 20,
    borderWidth: 2,
    shadow: true,
    raw: n.title || {}
  };
  if (n.shape === 'image' && n.image) {
    node.shape = 'image';
    node.image = n.image;
    node.size = n.size || 30;
    node.shapeProperties = { useBorderWithImage: true };
  } else if (/设备|Resource|Equipment/.test(group)) {
    node.shape = 'box';
  } else if (/Tool|刀具/.test(group)) {
    node.shape = 'diamond';
  } else if (/分析|Physics/.test(group)) {
    node.shape = 'triangleDown';
  } else {
    node.shape = 'dot';
  }
  return node;
}
function drawNeo4jVisNetwork(graph) {
  const container = $('mynetwork');
  if (!container) return;
  if (!window.vis || !vis.Network) {
    container.innerHTML = '<div class="graph-empty-box"><b>vis-network library was not loaded.</b><br>Please keep network access to unpkg.com or place vis-network.min.js locally.</div>';
    return;
  }
  destroyVisNetwork();
  const nodes = (graph.nodes || []).map(normalizeVisNode);
  const nodeIds = new Set(nodes.map(n => n.id));
  const edges = (graph.edges || []).filter(e => nodeIds.has(e.from) && nodeIds.has(e.to)).map(e => ({
    id: e.id, from: e.from, to: e.to, label: String(e.label || ''), arrows: 'to', title: typeof e.title === 'string' ? e.title : JSON.stringify(e.title || {})
  }));
  state.visNodes = new vis.DataSet(nodes);
  state.visEdges = new vis.DataSet(edges);
  const colorMap = neo4jColorMap();
  const options = {
    groups: {
      '零件': { color: colorMap['零件'], font: { size: 18, face: 'Microsoft YaHei' } },
      'Part': { color: colorMap['Part'], font: { size: 18, face: 'Microsoft YaHei' } },
      'Feature': { color: colorMap['Feature'], font: { size: 18, face: 'Microsoft YaHei' } },
      '工序': { color: colorMap['工序'] }, 'Operation': { color: colorMap['Operation'] },
      '工步': { color: colorMap['工步'], size: 15 }, 'Step': { color: colorMap['Step'], size: 15 }, 'Strategy': { color: colorMap['Strategy'], size: 15 },
      '设备': { color: colorMap['设备'], shape: 'box' }, 'Resource': { color: colorMap['Resource'], shape: 'box' }, 'Equipment': { color: colorMap['Equipment'], shape: 'box' },
      'Tool': { color: colorMap['Tool'], shape: 'diamond' }, '刀具': { color: colorMap['刀具'], shape: 'diamond' },
      '分析': { color: colorMap['分析'], shape: 'triangleDown' }, 'Physics Model': { color: colorMap['Physics Model'], shape: 'triangleDown' },
      'Image': { shape: 'image', color: colorMap['Image'], shapeProperties: { useBorderWithImage: true } },
      'Constraint': { color: colorMap['Constraint'] }
    },
    nodes: { shape: 'dot', size: 20, font: { size: 14, color: '#333', face: 'Microsoft YaHei, Arial' }, borderWidth: 2, shadow: true },
    edges: { width: 2, arrows: 'to', color: { color: '#848484', highlight: '#FF5722', hover: '#FF5722' }, smooth: { type: 'continuous' }, font: { size: 10, align: 'middle', color: '#555' } },
    physics: { enabled: true, barnesHut: { gravitationalConstant: -15000, springLength: 150, springConstant: 0.05 }, stabilization: { iterations: 1500 } },
    interaction: { hover: true, dragNodes: true, zoomView: true, dragView: true, navigationButtons: true, keyboard: true }
  };
  state.visNetwork = new vis.Network(container, { nodes: state.visNodes, edges: state.visEdges }, options);
  state.graph = { nodes, edges, index: Object.fromEntries(nodes.map(n => [String(n.id), n])) };
  state.visNetwork.on('stabilizationIterationsDone', () => state.visNetwork.setOptions({ physics: false }));
  state.visNetwork.on('click', params => { if (params.nodes && params.nodes.length) inspectNode(String(params.nodes[0])); });
}
function fitGraph() { if (state.visNetwork) state.visNetwork.fit({ animation: true }); }
function zoomGraph(factor) { if (!state.visNetwork) return; const pos = state.visNetwork.getViewPosition(); state.visNetwork.moveTo({ position: pos, scale: state.visNetwork.getScale() * factor, animation: true }); }
function featureFromNode(n) { const id = String(n.id || ''), label = String(n.label || ''); let m = id.match(/(?:feature|image|kg):([Ff]\d{2})/); if (m) return m[1].toUpperCase(); m = label.match(/\b([Ff]\d{2})\b/); return m ? m[1].toUpperCase() : ''; }
function inspectNode(id) {
  const n = state.graph?.index?.[String(id)] || (state.visNodes ? state.visNodes.get(id) : null); if (!n) return;
  const fid = featureFromNode(n); if (fid) { state.currentFeature = fid; updateCurrentFeatureBadges(); }
  let props = n.raw || {};
  if (typeof props === 'string') { try { props = JSON.parse(props); } catch { props = { title: props }; } }
  const summary = Object.entries(props).slice(0, 14).map(([k, v]) => `<div class="mini-kv"><b>${esc(k)}</b><span>${esc(Array.isArray(v) ? v.join('; ') : v)}</span></div>`).join('');
  const img = n.image ? `<img src="${esc(n.image)}">` : '';
  $('nodeInspector').innerHTML = `<div class="inspect-card"><b>${esc(n.label)}</b><span class="tag">${esc(n.group)}</span>${img}${summary || '<div class="hint">No properties on this Neo4j node.</div>'}</div>`;
}
async function submitNeo4jQuery(type, query) {
  if (!query) { alert('请输入查询内容或选择图片文件。'); return; }
  if ($('answer-box')) $('answer-box').textContent = '正在查询，请稍候...';
  try {
    const result = await api('/api/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ type, query }) });
    if ($('answer-box')) $('answer-box').textContent = result.answer || '';
    highlightGraph(result.highlight || {});
  } catch (err) {
    if ($('answer-box')) $('answer-box').textContent = '查询失败，请检查 Neo4j 连接和后端服务。';
  }
}
function highlightGraph(highlightData) {
  if (!state.visNodes || !state.visEdges || !state.visNetwork) return;
  const nodeIds = state.visNodes.getIds();
  state.visNodes.update(nodeIds.map(id => ({ id, color: null, size: null, font: { size: 14 } })));
  const edgeIds = state.visEdges.getIds();
  state.visEdges.update(edgeIds.map(id => ({ id, color: null, width: null })));
  if (highlightData && highlightData.nodes && highlightData.nodes.length > 0) {
    state.visNodes.update(highlightData.nodes.map(id => ({ id, color: { background: '#F44336', border: '#D32F2F' }, size: 30, font: { size: 18 } })));
    state.visEdges.update((highlightData.edges || []).map(id => ({ id, color: { color: '#F44336' }, width: 4 })));
    state.visNetwork.fit({ nodes: highlightData.nodes, animation: true });
  }
}
async function uploadDrawingFromInput(inputId) {
  const f = $(inputId)?.files?.[0]; if (!f) return;
  const form = new FormData(); form.append('file', f);
  const r = await fetch('/api/upload_view', { method: 'POST', body: form }).then(x => x.json());
  state.uploadedImage = r.path;
  if ($('visualReasoning')) $('visualReasoning').innerHTML += `<div class="kv"><b>Uploaded view</b><span>${esc(r.filename)}</span></div>`;
}
async function runVisualAnchor() {
  const q = $('text-query')?.value || '';
  const r = await api('/api/visual_anchor', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, image_file: state.uploadedImage, top_k: 5 }) });
  const matches = (r.top_matches || []).map(m => `${esc(m.feature_id)} (${Number(m.score || 0).toFixed(3)})`).join(', ');
  state.currentFeature = r.query_feature || state.currentFeature;
  state.detectedFeature = state.currentFeature;
  updateCurrentFeatureBadges();
  $('visualReasoning').innerHTML = kvs([['Anchored feature', r.query_feature], ['Anchor source', r.source || ''], ['Top-k matches', matches], ['Feature name', r.summary?.feature_name || ''], ['Constraints', (r.summary?.constraints || []).slice(0, 6).join('; ')]]);
}

async function loadPhysics() {
  if (!$('physicsStages')) return;
  updateCurrentFeatureBadges();
  if (!state.currentFeature) {
    $('physicsStages').innerHTML = '<div class="hint">No feature has been anchored yet. Run visual anchoring or inference first.</div>';
    $('physicsRules').innerHTML = '';
    $('physicsArchive').innerHTML = '<div class="hint">Waiting for current anchored feature.</div>';
    return;
  }
  const data = await api(`/api/physics_archive?feature_id=${encodeURIComponent(state.currentFeature)}`);
  $('physicsStages').innerHTML = (data.verification_stages || []).map((s, i) => `<div class="timeline-step"><div class="step-no">${i + 1}</div><div><b>${esc(s.name)}</b><p>${esc(s.detail)}</p><span class="status-pill ${esc(s.status)}">${esc(s.status)}</span></div></div>`).join('') || `<div class="hint">${esc(data.message || 'No physics archive record.')}</div>`;
  $('physicsRules').innerHTML = (data.rules || []).map(r => `<div class="rule-card"><b>${esc(r.rule_id)}</b><span>${esc(r.description)}</span><em>${esc(r.status)}</em></div>`).join('');
  if (data.feature?.feature_id) $('physicsArchive').innerHTML = kvs([['Feature', `${data.feature.feature_id} · ${data.feature.feature_name}`], ['Plan route', data.plan?.process_route || ''], ['Verification status', data.loop?.final_status || 'unchecked'], ['Retry count', data.loop?.retry_count ?? '-'], ['Control target', (data.feature.risk_tags || []).join(', ')]]);
}
async function loadEngine(loadTrace = false) {
  const status = await api('/api/engine_status');
  const cfg = status.finetuning_config || {};
  if ($('loraDataset')) $('loraDataset').value = cfg.dataset || 'data/processed/aero_instruct_5k.jsonl';
  if ($('loraRank')) $('loraRank').value = cfg.rank || 64;
  if ($('loraLr')) $('loraLr').value = cfg.learning_rate || '2e-5';
  if ($('loraMaxSteps')) $('loraMaxSteps').value = cfg.max_steps || 3900;
  if ($('loraCheckpoint')) $('loraCheckpoint').value = cfg.checkpoint_dir || 'adapters/aero_lora/checkpoints';
  if ($('loraMerged')) $('loraMerged').value = cfg.merged_model_dir || 'models/Qwen2.5-VL-72B-Instruct-aero-merged';
  if (loadTrace) await loadTrainingTrace(); else drawTrainingPlaceholder();
}
function drawTrainingPlaceholder() {
  const svg = $('trainingChart'); if (!svg) return;
  svg.innerHTML = `<rect x="16" y="18" width="1068" height="274" rx="8" fill="#f8fbff" stroke="#c7d6e8" stroke-dasharray="8 6"></rect><text x="550" y="145" text-anchor="middle" font-size="18" font-weight="800" fill="#26415f">No training trace loaded</text><text x="550" y="176" text-anchor="middle" font-size="13" fill="#60748b">Build a LoRA training command or click Load Training Trace after real training logs are available.</text>`;
}
async function loadTrainingTrace() {
  const trace = await api('/api/engine_training_trace');
  state.trainingTraceLoaded = true;
  drawTrainingChart(trace.records || []);
}
async function saveEngineConfig() {
  const payload = { dataset: $('loraDataset').value, rank: $('loraRank').value, learning_rate: $('loraLr').value, max_steps: $('loraMaxSteps').value, checkpoint_dir: $('loraCheckpoint').value, merged_model_dir: $('loraMerged').value };
  await api('/api/lora_config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  $('trainingCommandBox').textContent = 'Fine-tuning configuration saved. Build the training command when the local model path and dataset are ready.';
}
async function buildTrainingJob() {
  const payload = { dataset: $('loraDataset').value, rank: $('loraRank').value, learning_rate: $('loraLr').value, max_steps: $('loraMaxSteps').value, model_root: $('modelRoot')?.value, adapter_root: $('adapterRoot')?.value };
  const r = await api('/api/training_job', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  $('trainingCommandBox').textContent = [r.command, ...(r.warnings || []).map(w => 'WARNING: ' + w)].join('\n');
  drawTrainingPlaceholder();
}
function drawTrainingChart(rows) {
  const svg = $('trainingChart'); if (!svg) return; const W = 1100, H = 310, L = 62, R = 66, T = 28, B = 48;
  rows = rows.map(r => ({ step: +r.step, train_loss: +r.train_loss, val_loss: +r.val_loss, process_accuracy: +r.process_accuracy, visual_anchor_acc: +r.visual_anchor_acc, msrv: +r.msrv })).filter(r => Number.isFinite(r.step));
  if (!rows.length) { drawTrainingPlaceholder(); return; }
  const maxStep = Math.max(...rows.map(r => r.step), 1), maxLoss = Math.max(...rows.map(r => Math.max(r.train_loss, r.val_loss))), minLoss = Math.min(...rows.map(r => Math.min(r.train_loss, r.val_loss)));
  const x = v => L + (W - L - R) * (v / maxStep); const yLoss = v => T + (H - T - B) * (1 - ((v - minLoss) / (maxLoss - minLoss || 1))); const yPct = v => T + (H - T - B) * (1 - (v / 100));
  const poly = (key, y) => rows.map(r => `${x(r.step).toFixed(1)},${y(r[key]).toFixed(1)}`).join(' ');
  let grid = ''; for (let i = 0; i <= 5; i++) { const yy = T + i * (H - T - B) / 5, xx = L + i * (W - L - R) / 5; grid += `<line x1="${L}" y1="${yy}" x2="${W - R}" y2="${yy}" stroke="#d6e0ea"/><line x1="${xx}" y1="${T}" x2="${xx}" y2="${H - B}" stroke="#edf2f7"/><text x="10" y="${yy + 4}" font-size="11" fill="#64748b">${(maxLoss - i * (maxLoss - minLoss) / 5).toFixed(1)}</text><text x="${W - R + 8}" y="${yy + 4}" font-size="11" fill="#64748b">${(100 - i * 20).toFixed(0)}%</text>`; }
  const pts = (key, y, color) => rows.map(r => `<circle cx="${x(r.step).toFixed(1)}" cy="${y(r[key]).toFixed(1)}" r="3.2" fill="${color}"/>`).join('');
  svg.innerHTML = `${grid}<line x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}" stroke="#94a3b8"/><line x1="${L}" y1="${T}" x2="${L}" y2="${H - B}" stroke="#94a3b8"/><line x1="${W - R}" y1="${T}" x2="${W - R}" y2="${H - B}" stroke="#94a3b8"/><polyline points="${poly('train_loss', yLoss)}" fill="none" stroke="#ef4444" stroke-width="3"/><polyline points="${poly('val_loss', yLoss)}" fill="none" stroke="#f97316" stroke-width="2.4"/><polyline points="${poly('process_accuracy', yPct)}" fill="none" stroke="#3b82f6" stroke-width="3" stroke-dasharray="6 5"/><polyline points="${poly('msrv', yPct)}" fill="none" stroke="#16a34a" stroke-width="2.6" stroke-dasharray="3 5"/>${pts('train_loss', yLoss, '#ef4444')}${pts('process_accuracy', yPct, '#3b82f6')}<rect x="${W / 2 - 175}" y="7" width="350" height="20" rx="4" fill="#fff" stroke="#d6e0ea"/><text x="${W / 2 - 160}" y="22" font-size="12" fill="#ef4444">■ train loss</text><text x="${W / 2 - 65}" y="22" font-size="12" fill="#f97316">■ val loss</text><text x="${W / 2 + 20}" y="22" font-size="12" fill="#3b82f6">■ accuracy</text><text x="${W / 2 + 115}" y="22" font-size="12" fill="#16a34a">■ MSRV</text><text x="${W - R - 50}" y="${H - B + 30}" font-size="12" fill="#475569">step</text>`;
}

async function loadPrompts() {
  if (!$('promptEditor')) return;
  const data = await api('/api/prompt_templates');
  state.promptTemplates = data.templates || {};
  switchPromptTab(state.promptTab || 'system');
  renderPromptVariablesPending();
}
function switchPromptTab(tab) {
  state.promptTab = tab;
  document.querySelectorAll('.prompt-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  if ($('promptEditor')) $('promptEditor').value = state.promptTemplates[tab] || '';
  if ($('promptActiveStatus')) $('promptActiveStatus').textContent = `Editing: ${tab}`;
}
function renderPromptVariablesPending() {
  if (!$('promptVariables')) return;
  if (!state.currentFeature) {
    $('promptVariables').innerHTML = '<div class="hint">Waiting for visual anchoring / inference. Dynamic variables will be filled from the latest anchored feature and Aero-MPKG retrieval result.</div>';
    if ($('fusedPromptPreview')) $('fusedPromptPreview').textContent = 'Run visual anchoring or KGMCF inference first, then click Test Context Fusion.';
  }
}
async function savePrompt() {
  state.promptTemplates[state.promptTab] = $('promptEditor').value;
  const res = await api('/api/prompt_templates', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ templates: state.promptTemplates }) });
  state.promptTemplates = res.templates;
  $('fusedPromptPreview').textContent = 'Template saved.';
}
async function testContextFusion() {
  state.promptTemplates[state.promptTab] = $('promptEditor').value;
  if (!state.currentFeature) { renderPromptVariablesPending(); return; }
  const res = await api('/api/context_fusion', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ feature_id: state.currentFeature, intent: $('reasonIntent')?.value || 'Generate a process plan with knowledge-grounded constraints.' }) });
  $('promptVariables').innerHTML = Object.entries(res.variables || {}).map(([k, v]) => `<div class="var-card"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join('');
  $('fusedPromptPreview').textContent = res.fused_prompt || '';
}
async function loadRuntime() {
  const r = await api('/api/model_runtime');
  state.runtime = r;
  if ($('modelEndpoint')) $('modelEndpoint').value = r.endpoint_url || '';
  if ($('modelRoot')) $('modelRoot').value = r.model_root || '';
  if ($('adapterRoot')) $('adapterRoot').value = r.lora_adapter_root || '';
  if ($('remoteCallEnabled')) $('remoteCallEnabled').checked = !!r.call_enabled;
  renderRuntime(r);
  renderRuntimeSummary();
}
function renderRuntime(r) {
  if ($('runtimeStatus')) $('runtimeStatus').innerHTML = kvs([['Model', r.model_name || 'Qwen2.5-VL-72B-Instruct'], ['Endpoint', r.endpoint_url || '-'], ['Model path exists', r.model_path_exists], ['Adapter path exists', r.adapter_path_exists], ['Remote call enabled', r.call_enabled], ['Fallback mode', r.fallback_mode || '-']]);
}
function renderRuntimeSummary() {
  if (!$('runtimeSummary')) return;
  const r = state.runtime || {};
  $('runtimeSummary').innerHTML = kvs([['Runtime model', r.model_name || 'Qwen2.5-VL-72B-Instruct'], ['Endpoint', r.endpoint_url || '-'], ['Remote call', r.call_enabled ? 'enabled' : 'disabled; local KGMCF fallback available'], ['LoRA adapter', r.lora_adapter_root || '-']]);
}
async function saveRuntime() {
  const payload = { endpoint_url: $('modelEndpoint')?.value || state.runtime.endpoint_url, model_root: $('modelRoot')?.value || state.runtime.model_root, lora_adapter_root: $('adapterRoot')?.value || state.runtime.lora_adapter_root, enable_remote_call: !!$('remoteCallEnabled')?.checked };
  const r = await api('/api/model_runtime', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  state.runtime = r.runtime || state.runtime;
  renderRuntime(state.runtime); renderRuntimeSummary();
}
async function testRuntime() {
  const r = await api('/api/model_connection_test', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ endpoint_url: $('modelEndpoint')?.value || state.runtime.endpoint_url, api_key: $('modelApiKey')?.value || '' }) });
  if ($('runtimeStatus')) $('runtimeStatus').innerHTML += `<div class="kv"><b>Connection test</b><span>${esc(r.status)} · ${esc(r.message || r.endpoint_url || '')}</span></div>`;
}
async function uploadReasonDrawing() {
  const f = $('reasonUpload')?.files?.[0]; if (!f) return;
  const form = new FormData(); form.append('file', f);
  const r = await fetch('/api/upload_view', { method: 'POST', body: form }).then(x => x.json());
  state.reasonUploadedImage = r.path;
  $('reasonDrawingBox').innerHTML = `<img src="${imgUrl(r.path)}"><small>${esc(r.filename)} · uploaded. Run visual anchoring to detect feature category.</small>`;
  $('detectedFeatureBox').innerHTML = 'Detected feature: pending visual anchoring';
}
async function updateReasonDrawing() {
  if (!$('reasonDrawingBox')) return;
  if (state.reasonUploadedImage) {
    $('reasonDrawingBox').innerHTML = `<img src="${imgUrl(state.reasonUploadedImage)}"><small>Engineering drawing loaded. Run visual anchoring to detect feature category.</small>`;
    return;
  }
  $('reasonDrawingBox').innerHTML = '<span>Upload a 2D engineering view or open a sample from Dataset Management.</span>';
  if ($('detectedFeatureBox')) $('detectedFeatureBox').textContent = 'Detected feature: waiting for input';
}
async function anchorReasoningInput() {
  if (!state.reasonUploadedImage) { await updateReasonDrawing(); return null; }
  const r = await api('/api/visual_anchor', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: $('reasonIntent')?.value || '', image_file: state.reasonUploadedImage, top_k: 5 }) });
  state.currentFeature = r.query_feature || '';
  state.detectedFeature = state.currentFeature;
  updateCurrentFeatureBadges();
  const matches = (r.top_matches || []).map(m => `${m.feature_id} ${Number(m.score || 0).toFixed(3)}`).join(' | ');
  $('detectedFeatureBox').innerHTML = `Detected feature: <strong>${esc(state.currentFeature)}</strong> · ${esc(r.summary?.feature_name || '')}<br><small>${esc(matches)}</small>`;
  return r;
}
async function runQwenInference() {
  const anchor = await anchorReasoningInput();
  if (!anchor) { alert('Please upload an engineering drawing first.'); return; }
  const r = await api('/api/model_inference', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ feature_id: state.currentFeature, intent: $('reasonIntent').value, endpoint_url: $('modelEndpoint')?.value || state.runtime.endpoint_url, api_key: $('modelApiKey')?.value || '', image_file: state.reasonUploadedImage }) });
  state.currentWorkflow = r.fallback_workflow;
  state.currentFeature = r.feature_id || state.currentFeature;
  state.history.unshift(state.currentWorkflow);
  updateCurrentFeatureBadges();
  renderReasoningLog(state.currentWorkflow?.reasoning_log || [], r.model_result);
  renderReport(state.currentWorkflow?.process_card);
}
async function executeReasoning() {
  const anchor = await anchorReasoningInput();
  if (!anchor) { alert('Please upload an engineering drawing first.'); return; }
  state.currentMethod = $('reasonMethodSelect').value || 'KGMCF';
  const intent = $('reasonIntent').value || '';
  $('reasoningLog').innerHTML = '<div class="stage-name">[System] Running cognitive orchestration from uploaded engineering drawing...</div>';
  const r = await api('/api/run_workflow', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ feature_id: state.currentFeature, method: state.currentMethod, intent, max_retries: 3, image_file: state.reasonUploadedImage }) });
  state.currentWorkflow = r;
  state.currentFeature = r.feature?.feature_id || state.currentFeature;
  state.detectedFeature = state.currentFeature;
  state.history.unshift(r);
  updateCurrentFeatureBadges();
  renderRuntimeSummary();
  renderReasoningLog(r.reasoning_log || []);
  renderReport(r.process_card);
}
function renderReasoningLog(rows, modelResult) {
  let html = '';
  if (modelResult) html += `<div class="stage-name">[Qwen2.5-VL] ${esc(modelResult.status || 'runtime')}</div><div>${esc(modelResult.message || modelResult.endpoint_url || 'model response received')}</div>`;
  (rows || []).forEach(r => { const cls = r.stage === 'Verification' ? 'warn' : (r.stage === 'Output' ? 'ok' : 'stage-name'); html += `<div><span class="${cls}">[${esc(r.stage)}]</span> ${esc(r.message)}</div>`; });
  $('reasoningLog').innerHTML = html || '<div>No reasoning log.</div>';
}
async function loadReport() {
  if (!$('processReport')) return;
  updateCurrentFeatureBadges();
  if (state.currentWorkflow?.process_card) { renderReport(state.currentWorkflow.process_card); return; }
  $('processReport').innerHTML = '<div class="empty-report"><h2>No generated process card yet</h2><p>Open <b>Run Inference (CoT)</b>, upload an engineering drawing, run visual anchoring, and execute KGMCF cognitive orchestration. The process card will be generated from that latest workflow.</p></div>';
}
function renderReport(r) {
  if (!r || !$('processReport')) return;
  updateCurrentFeatureBadges();
  const f = r.feature || {}, steps = r.route_steps || [];
  $('processReport').innerHTML = `<div class="bar">SSF INFORMATION(异形特征信息)</div><div class="info-grid"><table class="info-table"><tr><td>FEATURE TYPE</td><td>${esc(f.feature_name || f.feature_id)}</td></tr><tr><td>PART NAME</td><td>${esc(r.part_name)}</td></tr><tr><td>SURFACE ROUGHNESS</td><td>${esc(r.surface_requirement)}</td></tr><tr><td>MACHINING PARAMETERS</td><td>${esc((r.constraints || []).slice(0, 4).join('; '))}</td></tr></table><div class="report-media">${r.image ? `<img src="${imgUrl(r.image)}">` : ''}<div><b>Feature ID</b><br>${esc(f.feature_id || '')}</div></div></div><div class="process-header">GENERATIVE PROCESS CARD (工艺过程卡片)</div><div class="process-meta"><div>Product: ${esc(r.part_name)}</div><div>Feature ID: ${esc(f.feature_id || '')}</div><div>Material: ${esc(r.material)}</div></div><div class="process-area"><div class="process-img">${r.image ? `<img src="${imgUrl(r.image)}">` : ''}<small>Target feature and geometric constraints</small></div><table class="process-table"><thead><tr><th>Step</th><th>Strategy</th><th>Resource / Tool</th><th>Control</th></tr></thead><tbody>${steps.map(s => `<tr class="${s.focus ? 'focus' : ''}"><td>${esc(s.step_no)}</td><td>${esc(s.operation)}</td><td>${esc(s.resource)}</td><td>${esc(s.control)}</td></tr>`).join('')}</tbody></table></div><div class="rationale"><b>Physics-Aware Cognitive Rationale:</b><br>${esc(r.rationale || 'The process plan was generated from Aero-MPKG evidence and symbolic verification rules.')}</div>`;
}
function downloadCurrentWorkflow() {
  if (!state.currentWorkflow) { alert('Please run inference first.'); return; }
  const blob = new Blob([JSON.stringify(state.currentWorkflow || {}, null, 2)], { type: 'application/json' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'kgmcf_workflow_record.json'; a.click(); URL.revokeObjectURL(a.href);
}
function downloadProcessPdf() {
  if (!state.currentWorkflow) { alert('Please run inference first. The PDF is exported from the generated workflow.'); return; }
  const fid = state.currentWorkflow.feature?.feature_id || state.currentFeature || '';
  window.location = `/api/process_card_pdf?feature_id=${encodeURIComponent(fid)}&method=${encodeURIComponent(state.currentWorkflow.method || 'KGMCF')}`;
}
function renderHistory() {
  if (!$('historyList')) return;
  $('historyList').innerHTML = state.history.map((h, i) => `<div class="history-item"><b>${i + 1}. ${esc(h.feature?.feature_id)} · ${esc(h.method)}</b><p>${esc(h.intent || '')}</p><small>${new Date((h.timestamp || 0) * 1000).toLocaleString()}</small></div>`).join('') || '<div class="empty">No local workflow history yet.</div>';
}
async function renderBackendStatus(id) {
  if (id === 'neo4jStatus') { await loadNeo4jStatus(); return; }
  const s = await api('/api/engine_status');
  $(id).innerHTML = kvs([['Model interface', s.model_interface], ['Active backend', s.active_backend], ['Visual encoder', s.visual_encoder], ['Knowledge backend', s.knowledge_backend], ['Endpoint', s.runtime?.endpoint_url || '-'], ['Model root', s.runtime?.model_root || '-'], ['Adapter root', s.runtime?.lora_adapter_root || '-']]);
}

init().catch(err => { console.error(err); document.body.insertAdjacentHTML('afterbegin', `<div style="background:#fee;padding:12px;color:#900">${esc(err.message)}</div>`); });
