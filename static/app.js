(function(){
  const $ = (sel) => document.querySelector(sel);

  // Elements
  const btnPredict = $('#btnPredict');
  const btnClear   = $('#btnClear');
  const txt        = $('#text');

  const fileInput  = $('#fileInput');
  const btnUpload  = $('#btnUpload');

  const resultsCard = $('#results_card');
  const resultLine  = $('#result_line');
  const hiWrap      = $('#highlighted_wrap');
  const hiText      = $('#highlighted_text');
  const spansWrap   = $('#spans_wrap');
  const spansBody   = document.querySelector('#spans_table tbody');

  const batchCard   = $('#batch_card');
  const batchBody   = document.querySelector('#batch_table tbody');

  // Controls
  const thSlider    = $('#thSlider');
  const thVal       = $('#thVal');
  const redactToggle= $('#redactToggle');

  // Export
  const btnExportCSV  = $('#btnExportCSV');
  const btnExportDocx = $('#btnExportDocx');

  // Charts
  let probChart = null;
  let batchChart = null;

  // Keep last data to re-render on toggle/threshold change
  let lastSingleData = null;
  let lastSingleInput = '';
  let lastBatchData = null;

  // Colors
  const ORANGE = '#f36f21'; // xúc phạm
  const BLUE   = '#2e3192'; // không

  // INIT threshold from localStorage
  const savedTh = Number(localStorage.getItem('threshold') || 50);
  thSlider.value = isFinite(savedTh) ? savedTh : 50;
  thVal.textContent = thSlider.value;

  function setThreshold(v){
    thVal.textContent = v;
    localStorage.setItem('threshold', v);
  }

  thSlider.addEventListener('input', ()=>{
    setThreshold(thSlider.value);
    if(lastSingleData){ renderSingle(lastSingleInput, lastSingleData); }
    if(lastBatchData){ renderBatch(lastBatchData); }
  });

  redactToggle.addEventListener('change', ()=>{
    if(lastSingleData){ renderSingle(lastSingleInput, lastSingleData); }
    if(lastBatchData){ renderBatch(lastBatchData); }
  });

  function destroyChart(chart){ if(chart){ chart.destroy(); } }

  function doughnut(ctx, values){
    return new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Xúc phạm', 'Không'],
        datasets: [{
          data: values,
          backgroundColor: [ORANGE, BLUE],
          borderColor: ['#fff', '#fff'],
          borderWidth: 1,
          cutout: '60%'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        layout: { padding: 4 }
      }
    });
  }

  function styleMarks(containerEl, prob){
    const p = Math.max(0, Math.min(Number(prob)||0, 100));
    const alpha = 0.3 + 0.6 * (p/100); // 0.3 → 0.9
    containerEl.querySelectorAll('mark').forEach(m => {
      m.style.backgroundColor = `rgba(243,111,33,${alpha})`; // cam theo xác suất
      m.style.color = '#000';
    });
  }

  function escapeHtml(s){
    return (s || '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  }

  function redactText(original, spans){
    if(!Array.isArray(spans) || spans.length===0) return escapeHtml(original);
    let out = '', last = 0;
    const sorted = [...spans].sort((a,b)=> a.start - b.start);
    sorted.forEach(s=>{
      out += escapeHtml(original.slice(last, s.start));
      out += '***';
      last = s.end;
    });
    out += escapeHtml(original.slice(last));
    return out;
  }

  // ----- Single rendering -----
  function renderSingle(inputText, data){
    const threshold = Number(thSlider.value);
    const prob = (typeof data.probability_profane === 'number')
      ? Number(data.probability_profane) : (data.prediction ? 100 : 0);

    const effPred = (prob >= threshold) || data.is_profane_by_list || (Array.isArray(data.spans) && data.spans.length>0);

    const verdict = effPred ? 'Có dấu hiệu độc hại/tiêu cực' : 'Không phát hiện độc hại/tiêu cực';
    const byList = data.is_profane_by_list ? ' — Khớp từ điển' : '';
    resultLine.textContent = `${verdict} (xác suất: ${prob.toFixed(2)}%, ngưỡng: ${threshold}%)${byList}`;

    // Redact / Highlight
    if (redactToggle.checked) {
      hiText.innerHTML = redactText(inputText, data.spans || []);
    } else {
      hiText.innerHTML = data.highlighted_html || escapeHtml(inputText);
      styleMarks(hiText, prob);
    }
    hiWrap.style.display = 'block';

    // Spans table
    spansBody.innerHTML = '';
    if (Array.isArray(data.spans) && data.spans.length > 0) {
      data.spans.forEach(s => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${s.text}</td>
          <td>[${s.start}, ${s.end})</td>
          <td>${(s.source || []).join(', ')}</td>`;
        spansBody.appendChild(tr);
      });
      spansWrap.style.display = 'block';
    } else {
      spansWrap.style.display = 'none';
    }

    // Doughnut single
    const clean = Math.max(0, 100 - prob);
    destroyChart(probChart);
    probChart = doughnut(document.getElementById('probChart').getContext('2d'), [prob, clean]);

    resultsCard.style.display = 'block';
  }

  // ----- Batch rendering -----
  function renderBatch(data){
    const items = Array.isArray(data.items) ? data.items : [];
    lastBatchData = data;

    // Count for doughnut (theo ngưỡng + span)
    const threshold = Number(thSlider.value);
    let off = 0, clean = 0;
    items.forEach(it => {
      const p = (typeof it.probability_profane === 'number') ? Number(it.probability_profane) : (it.prediction ? 100 : 0);
      const eff = (p >= threshold) || it.is_profane_by_list || (Array.isArray(it.spans) && it.spans.length>0);
      if (eff) off++; else clean++;
    });
    destroyChart(batchChart);
    batchChart = doughnut(document.getElementById('batchChart').getContext('2d'), [off, clean]);

    // Bảng tổng hợp — hiển thị TỐI ĐA 200 dòng (server cũng đã giới hạn 200)
    batchBody.innerHTML = '';
    items.forEach(it => {
      const p = (typeof it.probability_profane === 'number') ? Number(it.probability_profane) : (it.prediction ? 100 : 0);
      const eff = (p >= threshold) || it.is_profane_by_list || (Array.isArray(it.spans) && it.spans.length>0);
      const tr = document.createElement('tr');
      const textCell = document.createElement('td');

      if (redactToggle.checked) {
        textCell.innerHTML = redactText(it.text, it.spans || []);
      } else {
        textCell.innerHTML = it.highlighted_html || escapeHtml(it.text);
        styleMarks(textCell, p);
      }

      tr.innerHTML = `
        <td>${it.index + 1}</td>
        <td>${p}</td>
        <td>${eff ? 'Độc hại' : 'Không độc hại'}</td>
      `;
      tr.appendChild(textCell);
      batchBody.appendChild(tr);
    });

    batchCard.style.display = 'block';
  }

  // ----- Events -----
  btnPredict.addEventListener('click', async () => {
    const val = (txt.value || '').trim();
    if(!val){ alert('Hãy nhập đoạn phản hồi trước nhé!'); return; }

    lastSingleInput = val;
    lastSingleData = null;
    resultLine.textContent = 'Đang phân tích...';
    resultsCard.style.display = 'block';

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text: val })
      });
      const data = await res.json();
      if(data.error){ throw new Error(data.error); }

      lastSingleData = data;
      renderSingle(val, data);
      // clear batch area
      batchCard.style.display = 'none';
    } catch (e) {
      resultLine.textContent = 'Lỗi: ' + (e.message || 'không xác định');
      hiWrap.style.display = 'none';
      spansWrap.style.display = 'none';
    }
  });

  btnClear.addEventListener('click', () => {
    txt.value = '';
    lastSingleData = null;
    resultsCard.style.display = 'none';
  });

  btnUpload.addEventListener('click', async () => {
    const file = fileInput.files && fileInput.files[0];
    if(!file){ alert('Hãy chọn file (CSV/TXT/XLSX/DOCX/PDF) trước nhé!'); return; }

    // reset single
    resultsCard.style.display = 'none';
    lastBatchData = null;

    const form = new FormData();
    form.append('file', file);

    try{
      const res = await fetch('/api/upload', { method: 'POST', body: form });
      const data = await res.json();
      if(data.error){ throw new Error(data.error); }

      lastBatchData = data;
      renderBatch(data);
    } catch (e) {
      alert('Lỗi: ' + (e.message || 'không xác định'));
      batchCard.style.display = 'none';
    }
  });

  // ----- Export CSV (client-side) -----
  if (btnExportCSV) {
    btnExportCSV.addEventListener('click', ()=>{
      if(!lastBatchData || !Array.isArray(lastBatchData.items)){ return; }
      const rows = [['index','probability','prediction','text','spans_text','spans_pos']];
      lastBatchData.items.forEach(it=>{
        const spansText = (it.spans||[]).map(s=>s.text).join('|');
        const spansPos  = (it.spans||[]).map(s=>`[${s.start},${s.end})`).join('|');
        rows.push([it.index+1, it.probability_profane ?? '', it.prediction, it.text, spansText, spansPos]);
      });
      const csv = rows.map(r => r.map(v => `"${String(v).replace(/"/g,'""')}"`).join(',')).join('\n');
      const blob = new Blob([csv], {type:'text/csv;charset=utf-8;'});
      const url  = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'ket_qua_phan_tich.csv'; a.click();
      URL.revokeObjectURL(url);
    });
  }

  // ----- Export DOCX (server-side) -----
  if (btnExportDocx) {
    btnExportDocx.addEventListener('click', async ()=>{
      if(!lastBatchData || !Array.isArray(lastBatchData.items)){ return; }
      try{
        const res = await fetch('/api/export_docx', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ items: lastBatchData.items })
        });
        if(!res.ok){ throw new Error('Xuất DOCX thất bại'); }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'bao_cao_highlight.docx'; a.click();
        URL.revokeObjectURL(url);
      }catch(e){
        alert(e.message || 'Không thể xuất DOCX');
      }
    });
  }
})();
