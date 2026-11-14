document.addEventListener('DOMContentLoaded', async () => {
  const ctxRoc = document.getElementById('rocChart').getContext('2d');
  const ctxConf = document.getElementById('confMatrix').getContext('2d');
  const ctxFeat = document.getElementById('featImpChart').getContext('2d');

  // Giá trị mặc định phòng trường hợp chưa chạy pipeline
  let stats = {
    total_students: 0,
    dropout_rate: 0,
    best_model: 'Chưa huấn luyện',
    roc_curve: {
      labels: ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
      values: [0, 0.55, 0.78, 0.88, 0.95, 1]
    },
    confusion_matrix: { tn: 50, fp: 5, fn: 10, tp: 35 },
    feature_importance: {
      labels: ['Curricular units 2nd sem (approved)', 'Tuition fees up to date', 'Curricular units 1st sem (approved)', 'Age at enrollment', 'Debtor'],
      values: [0.35, 0.25, 0.20, 0.12, 0.08]
    }
  };

  try {
    const res = await fetch('/stats');
    if (res.ok) {
      const apiData = await res.json();
      stats = { ...stats, ...apiData };
    }
  } catch (err) {
    console.warn('Không kết nối được /stats, dùng dữ liệu mẫu:', err);
  }

  document.getElementById('kpiTotal').textContent = stats.total_students;
  document.getElementById('kpiRate').textContent = `${stats.dropout_rate}%`;
  document.getElementById('kpiModel').textContent = stats.best_model;

  new Chart(ctxRoc, {
    type: 'line',
    data: {
      labels: stats.roc_curve.labels,
      datasets: [{
        label: 'ROC Curve',
        data: stats.roc_curve.values,
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.2
      }]
    }
  });

  const cm = stats.confusion_matrix;
  new Chart(ctxConf, {
    type: 'bar',
    data: {
      labels: ['Thực tế: Graduate (0)', 'Thực tế: Dropout (1)'],
      datasets: [
        {
          label: 'Dự đoán: Graduate (0)',
          data: [cm.tn, cm.fn],
          backgroundColor: 'rgba(54, 162, 235, 0.6)'
        },
        {
          label: 'Dự đoán: Dropout (1)',
          data: [cm.fp, cm.tp],
          backgroundColor: 'rgba(255, 99, 132, 0.6)'
        }
      ]
    },
    options: { indexAxis: 'y' }
  });

  new Chart(ctxFeat, {
    type: 'bar',
    data: {
      labels: stats.feature_importance.labels.length ? stats.feature_importance.labels : ['N/A'],
      datasets: [{
        label: 'Importance',
        data: stats.feature_importance.values.length ? stats.feature_importance.values : [0],
        backgroundColor: 'rgba(153, 102, 255, 0.6)'
      }]
    },
    options: { indexAxis: 'y' }
  });

  // Xử lý form dự đoán nhanh
  const form = document.getElementById('predictForm');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const resultDiv = document.getElementById('predictResult');
    resultDiv.textContent = 'Đang tính toán...';

    const payload = {
      features: {
        'Curricular units 1st sem (approved)': Number(document.getElementById('sem1Approved').value),
        'Curricular units 2nd sem (approved)': Number(document.getElementById('sem2Approved').value),
        'Tuition fees up to date': Number(document.getElementById('tuitionUpToDate').value)
      }
    };

    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      if (!resp.ok) {
        resultDiv.textContent = `Lỗi: ${data.detail || 'Không thể dự đoán'}`;
        return;
      }
      resultDiv.textContent = `Xác suất bỏ học: ${(data.dropout_probability * 100).toFixed(2)}% | Mức rủi ro: ${data.risk_tier}`;
    } catch (err) {
      resultDiv.textContent = 'Lỗi kết nối tới API server.';
    }
  });

  // Theme toggle
  const btn = document.getElementById('themeToggle');
  btn.addEventListener('click', () => {
    document.body.classList.toggle('dark');
    document.querySelector('header').classList.toggle('dark');
  });
});