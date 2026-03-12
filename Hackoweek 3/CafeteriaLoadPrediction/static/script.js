document.addEventListener('DOMContentLoaded', () => {
    const ctx = document.getElementById('loadChart').getContext('2d');

    // Set default Chart.js font and color
    Chart.defaults.color = '#8b949e';
    Chart.defaults.font.family = "'Inter', sans-serif";

    const loadChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Simulated Reality',
                    data: [],
                    borderColor: '#3fb950',
                    backgroundColor: 'rgba(63, 185, 80, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#3fb950',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#3fb950',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: false,
                    tension: 0.4 // Smooth curve
                },
                {
                    label: 'Model Predicted Load',
                    data: [],
                    borderColor: '#ff7b72',
                    backgroundColor: 'rgba(255, 123, 114, 0.1)',
                    borderWidth: 3,
                    borderDash: [6, 6],
                    pointBackgroundColor: '#ff7b72',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#ff7b72',
                    pointRadius: 0, // Show points only on hover
                    pointHoverRadius: 6,
                    fill: false,
                    tension: 0.4 // Smooth curve
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        boxWidth: 8,
                        padding: 20,
                        font: {
                            size: 13
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(22, 27, 34, 0.95)',
                    titleColor: '#f0f6fc',
                    bodyColor: '#c9d1d9',
                    borderColor: 'rgba(48, 54, 61, 0.8)',
                    borderWidth: 1,
                    padding: 14,
                    titleFont: {
                        size: 14,
                        weight: 'bold',
                        family: "'Inter', sans-serif"
                    },
                    bodyFont: {
                        family: "'Inter', sans-serif",
                        size: 13
                    },
                    bodySpacing: 8,
                    usePointStyle: true,
                    boxPadding: 6
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(48, 54, 61, 0.2)',
                        drawBorder: false,
                    },
                    ticks: {
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 12,
                        font: {
                            family: "'Inter', sans-serif"
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(48, 54, 61, 0.2)',
                        drawBorder: false,
                    },
                    title: {
                        display: true,
                        text: 'Load (People)',
                        color: '#8b949e',
                        font: {
                            family: "'Inter', sans-serif",
                            size: 13
                        }
                    },
                    ticks: {
                        font: {
                            family: "'Inter', sans-serif"
                        }
                    }
                }
            },
            animation: {
                duration: 500, // Smooth transition for new data points
                easing: 'easeOutExpo'
            }
        }
    });

    const maxDataPoints = 35; // Number of points to display

    function updateDashboard(data) {
        // Update Chart
        loadChart.data.labels.push(data.timestamp);
        loadChart.data.datasets[0].data.push(data.actual_load);
        loadChart.data.datasets[1].data.push(data.predicted_load);

        if (loadChart.data.labels.length > maxDataPoints) {
            loadChart.data.labels.shift();
            loadChart.data.datasets[0].data.shift();
            loadChart.data.datasets[1].data.shift();
        }

        loadChart.update();

        // Update cards
        const tempEl = document.getElementById('current-temp');
        const predEl = document.getElementById('current-predicted');

        tempEl.textContent = `${data.temperature}°C`;

        // Add animation class
        predEl.textContent = data.predicted_load;
        predEl.style.transform = 'scale(1.15)';
        predEl.style.color = '#a5d6ff'; // glow
        setTimeout(() => {
            predEl.style.transform = 'scale(1)';
            predEl.style.color = 'var(--accent)';
        }, 200);
    }

    // Handle WebSocket
    const statusEl = document.getElementById('connection-status');
    const wsUrl = `ws://${window.location.host}/ws`;

    function connect() {
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            statusEl.textContent = 'Connected & Streaming';
            statusEl.className = 'status-connected';
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);

            if (message.type === 'history') {
                // Bulk load history
                message.data.forEach(item => {
                    loadChart.data.labels.push(item.timestamp);
                    loadChart.data.datasets[0].data.push(item.actual_load);
                    loadChart.data.datasets[1].data.push(item.predicted_load);
                });
                loadChart.update();

                // Update cards with latest from history
                if (message.data.length > 0) {
                    const latest = message.data[message.data.length - 1];
                    document.getElementById('current-temp').textContent = `${latest.temperature}°C`;
                    document.getElementById('current-predicted').textContent = latest.predicted_load;
                }
            } else if (message.type === 'update') {
                updateDashboard(message.data);
            }
        };

        ws.onclose = () => {
            statusEl.textContent = 'Connection Lost. Reconnecting...';
            statusEl.className = 'status-connecting';
            setTimeout(connect, 3000);
        };

        ws.onerror = (err) => {
            console.error('WebSocket encountered error: ', err, 'Closing socket');
            ws.close();
        };
    }

    // Start WebSocket connection
    connect();
});
