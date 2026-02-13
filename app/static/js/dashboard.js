/**
 * ═══════════════════════════════════════════════════════════════════
 * YANTRA CENTRAL — Dashboard Engine
 * Handles data fetching, chart rendering, map, and real-time updates
 * ═══════════════════════════════════════════════════════════════════
 */

// ─── CONFIGURATION ──────────────────────────────────────────────────
const API_BASE = '';  // Same origin; change to 'http://localhost:8000' if separated
const REFRESH_INTERVAL = 8000; // ms between data refreshes

// ─── STATE ──────────────────────────────────────────────────────────
let charts = {};
let cityMap = null;
let mapMarkers = [];
let sidebarCollapsed = false;
let currentPage = 'command-center';

const PAGE_TITLES = {
    'command-center': 'Command Center <span>/ Infrastructure Overview</span>',
    'drainage': 'Drainage <span>/ Network Monitoring</span>',
    'stress-score': 'Drainage Stress <span>/ ML Analytics</span>',
    'analytics': 'Analytics <span>/ System-Wide Trends</span>',
    'roads': 'Roads <span>/ Infrastructure Health</span>',
    'bridges': 'Bridges <span>/ Structural Health</span>',
    'buildings': 'Buildings <span>/ Safety Monitoring</span>',
    'ai-insights': 'AI Insights <span>/ Intelligence Hub</span>',
    'settings': 'Settings <span>/ System Configuration</span>',
};

// ─── API SERVICE LAYER ──────────────────────────────────────────────
const api = {
    async get(endpoint) {
        try {
            const res = await fetch(`${API_BASE}${endpoint}`);
            if (!res.ok) throw new Error(`API error: ${res.status}`);
            return await res.json();
        } catch (err) {
            console.error(`Failed to fetch ${endpoint}:`, err);
            return null;
        }
    },
    getKPIs: () => api.get('/api/kpis'),
    getSensorData: (type) => api.get(`/api/sensors/${type}`),
    getNodes: () => api.get('/api/nodes'),
    getZones: () => api.get('/api/zones'),
    getAlerts: () => api.get('/api/alerts'),
    getRecommendations: () => api.get('/api/recommendations'),
    getInsights: () => api.get('/api/insights'),
    getSystemStatus: () => api.get('/api/system-status'),
    getStressScore: () => api.get('/api/stress-score'),
    getStressHistory: () => api.get('/api/stress-score/history'),
    getRoadHealth: () => api.get('/api/road-health'),
    getBridgeHealth: () => api.get('/api/bridge-health'),
};

// ─── INITIALIZATION ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    initLucideIcons();
    initSidebar();
    initClock();
    initMap();
    await loadAllData();
    startAutoRefresh();
});

function initLucideIcons() {
    if (window.lucide) {
        lucide.createIcons();
    }
}

// ─── SIDEBAR + PAGE NAVIGATION ──────────────────────────────────────
function initSidebar() {
    const toggle = document.getElementById('sidebar-toggle');
    const sidebar = document.getElementById('sidebar');
    if (toggle && sidebar) {
        toggle.addEventListener('click', () => {
            sidebarCollapsed = !sidebarCollapsed;
            sidebar.classList.toggle('collapsed', sidebarCollapsed);
        });
    }

    // Nav items — full page switching
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const page = item.getAttribute('data-page');
            if (page) navigateToPage(page);
        });
    });
}

function navigateToPage(page) {
    if (page === currentPage) return;
    currentPage = page;

    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const activeNav = document.querySelector(`.nav-item[data-page="${page}"]`);
    if (activeNav) activeNav.classList.add('active');

    // Switch visible page section
    document.querySelectorAll('.page-section').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(`page-${page}`);
    if (target) target.classList.add('active');

    // Update header title
    const titleEl = document.querySelector('.header-title');
    if (titleEl && PAGE_TITLES[page]) {
        titleEl.innerHTML = PAGE_TITLES[page];
    }

    // Refresh icons for the new page
    initLucideIcons();

    // Fix Leaflet map render when switching back to command center
    if (page === 'command-center' && cityMap) {
        setTimeout(() => cityMap.invalidateSize(), 100);
    }

    // If switching to Analytics, render charts if not already done
    if (page === 'analytics') initAnalyticsCharts();

    // Load data for specific pages
    if (page === 'stress-score') loadStressScorePage();
    if (page === 'roads') loadRoadHealthPage();
    if (page === 'bridges') loadBridgeHealthPage();

    // Scroll to top of content
    const main = document.getElementById('main-content');
    if (main) main.scrollTop = 0;
}

// ─── LIVE CLOCK ─────────────────────────────────────────────────────
function initClock() {
    const el = document.getElementById('live-clock');
    function tick() {
        const now = new Date();
        const opts = { weekday: 'short', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
        el.textContent = now.toLocaleDateString('en-IN', opts);
    }
    tick();
    setInterval(tick, 1000);
}

// ─── LOAD ALL DATA ──────────────────────────────────────────────────
async function loadAllData() {
    const [kpis, rainfall, water, soil, nodes, alerts, recommendations, insights, sysStatus, stressScore, roadHealth, bridgeHealth] = await Promise.all([
        api.getKPIs(),
        api.getSensorData('rainfall'),
        api.getSensorData('water_level'),
        api.getSensorData('soil_moisture'),
        api.getNodes(),
        api.getAlerts(),
        api.getRecommendations(),
        api.getInsights(),
        api.getSystemStatus(),
        api.getStressScore(),
        api.getRoadHealth(),
        api.getBridgeHealth(),
    ]);

    if (kpis) renderKPIs(kpis);
    if (rainfall) renderChart('rainfall-chart', rainfall, 'Rainfall', '#06d6a0', 'rgba(6,214,160,0.1)');
    if (water) renderChart('water-chart', water, 'Water Level', '#3b82f6', 'rgba(59,130,246,0.1)', 4.2);
    if (soil) renderChart('soil-chart', soil, 'Soil Moisture', '#a855f7', 'rgba(168,85,247,0.1)');
    if (nodes) { renderDrainageTable(nodes); updateMapMarkers(nodes); }
    if (alerts) renderAlerts(alerts);
    if (recommendations) renderRepairs(recommendations);
    if (insights) renderInsights(insights);
    if (sysStatus) updateSystemStatus(sysStatus);

    // Render module cards on command center
    renderModuleCards(stressScore, roadHealth, bridgeHealth);

    initLucideIcons();
}

// ─── AUTO REFRESH ───────────────────────────────────────────────────
function startAutoRefresh() {
    setInterval(async () => {
        const [kpis, nodes, alerts] = await Promise.all([
            api.getKPIs(),
            api.getNodes(),
            api.getAlerts(),
        ]);
        if (kpis) renderKPIs(kpis);
        if (nodes) { renderDrainageTable(nodes); updateMapMarkers(nodes); }
        if (alerts) renderAlerts(alerts);
    }, REFRESH_INTERVAL);

    // Refresh charts less frequently
    setInterval(async () => {
        const [rainfall, water, soil] = await Promise.all([
            api.getSensorData('rainfall'),
            api.getSensorData('water_level'),
            api.getSensorData('soil_moisture'),
        ]);
        if (rainfall) updateChartData('rainfall-chart', rainfall);
        if (water) updateChartData('water-chart', water);
        if (soil) updateChartData('soil-chart', soil);
    }, REFRESH_INTERVAL * 2);
}

// ─── MODULE CARDS (Command Center) ─────────────────────────────────
function renderModuleCards(stressData, roadData, bridgeData) {
    const container = document.getElementById('module-cards-grid');
    if (!container) return;

    const stressScore = stressData?.today?.score ?? '--';
    const stressTrend = stressData?.trend ?? 'stable';
    const roadScore = roadData?.summary?.overall_health ?? '--';
    const roadTrend = roadData?.summary?.trend ?? 'stable';
    const bridgeScore = bridgeData?.summary?.overall_health ?? '--';
    const bridgeTrend = bridgeData?.summary?.trend ?? 'stable';

    function getScoreColor(score) {
        if (score === '--') return 'var(--text-muted)';
        if (score > 70) return 'var(--emerald)';
        if (score > 40) return 'var(--amber)';
        return 'var(--red)';
    }

    function getTrendIcon(trend) {
        if (trend === 'rising' || trend === 'declining') return '▲';
        if (trend === 'falling' || trend === 'improving') return '▼';
        return '●';
    }

    function getTrendColor(trend) {
        if (trend === 'rising' || trend === 'declining') return 'var(--red)';
        if (trend === 'falling' || trend === 'improving') return 'var(--emerald)';
        return 'var(--text-muted)';
    }

    const modules = [
        {
            id: 'stress-score',
            icon: 'activity',
            name: 'Drainage Stress',
            score: stressScore,
            unit: '/100',
            trend: stressTrend,
            color: 'cyan',
            desc: 'ML-predicted stress index',
            badge: stressData?.model_info?.name ? 'ML Model' : 'Mock',
        },
        {
            id: 'roads',
            icon: 'road',
            name: 'Road Health',
            score: roadScore,
            unit: '%',
            trend: roadTrend,
            color: 'amber',
            desc: `${roadData?.summary?.total_potholes ?? 0} potholes detected`,
            badge: `${roadData?.summary?.critical_segments ?? 0} critical`,
        },
        {
            id: 'bridges',
            icon: 'bridge',
            name: 'Bridge Health',
            score: bridgeScore,
            unit: '%',
            trend: bridgeTrend,
            color: 'purple',
            desc: `${bridgeData?.summary?.total_bridges ?? 0} bridges monitored`,
            badge: `${bridgeData?.summary?.bridges_at_risk ?? 0} at risk`,
        },
        {
            id: 'buildings',
            icon: 'building-2',
            name: 'Building Safety',
            score: '--',
            unit: '',
            trend: 'stable',
            color: 'blue',
            desc: 'Module under development',
            badge: '🚧 Coming Soon',
        },
    ];

    container.innerHTML = modules.map((mod, i) => `
        <div class="module-card animate-in ${mod.id === 'buildings' ? 'disabled' : ''}" data-page="${mod.id}" style="animation-delay:${i * 0.08}s">
            <div class="module-card-glow ${mod.color}"></div>
            <div class="module-card-header">
                <div class="module-card-icon ${mod.color}">
                    <i data-lucide="${mod.icon}" style="width:24px;height:24px"></i>
                </div>
                <span class="module-card-badge ${mod.color}">${mod.badge}</span>
            </div>
            <div class="module-card-name">${mod.name}</div>
            <div class="module-card-score" style="color:${getScoreColor(mod.score)}">
                ${mod.score}<span class="module-card-unit">${mod.unit}</span>
            </div>
            <div class="module-card-footer">
                <span class="module-card-desc">${mod.desc}</span>
                <span class="module-card-trend" style="color:${getTrendColor(mod.trend)}">${getTrendIcon(mod.trend)} ${mod.trend}</span>
            </div>
            ${mod.id !== 'buildings' ? '<div class="module-card-cta">View Details →</div>' : ''}
        </div>
    `).join('');

    // Add click handlers
    container.querySelectorAll('.module-card:not(.disabled)').forEach(card => {
        card.addEventListener('click', () => {
            const page = card.getAttribute('data-page');
            if (page) navigateToPage(page);
        });
    });

    initLucideIcons();
}

// ─── KPI CARDS ──────────────────────────────────────────────────────
function renderKPIs(data) {
    const container = document.getElementById('kpi-row');
    const kpiConfigs = [
        { key: 'active_sensors', label: 'Active Sensors', icon: 'radio-tower', color: 'cyan', format: v => `${v.value}/${v.total}`, meta: v => `${v.uptime}% uptime` },
        { key: 'flood_risk_index', label: 'Flood Risk Index', icon: 'alert-triangle', color: 'amber', format: v => v.value, meta: v => `Trend: ${v.trend}` },
        { key: 'avg_rainfall', label: 'Avg Rainfall', icon: 'cloud-rain', color: 'blue', format: v => `${v.value}`, meta: v => v.unit },
        { key: 'critical_alerts', label: 'Critical Alerts', icon: 'siren', color: 'red', format: v => v.value, meta: v => `${v.unacknowledged} unacknowledged` },
        { key: 'infra_health', label: 'Infra Health Score', icon: 'heart-pulse', color: 'purple', format: v => `${v.value}%`, meta: v => `Trend: ${v.trend}` },
    ];

    container.innerHTML = kpiConfigs.map((cfg, i) => {
        const v = data[cfg.key];
        if (!v) return '';
        const trendClass = v.trend === 'up' ? 'up' : v.trend === 'down' ? 'down' : 'stable';
        const trendArrow = v.trend === 'up' ? '▲' : v.trend === 'down' ? '▼' : '●';
        return `
            <div class="card kpi-card animate-in" style="animation-delay:${i * 0.05}s">
                <div class="kpi-glow ${cfg.color}"></div>
                <div class="kpi-icon ${cfg.color}">
                    <i data-lucide="${cfg.icon}" style="width:22px;height:22px"></i>
                </div>
                <div class="kpi-label">${cfg.label}</div>
                <div class="kpi-value" style="color:var(--${cfg.color === 'purple' ? 'purple' : cfg.color})">${cfg.format(v)}</div>
                <div class="kpi-meta">
                    <span class="kpi-trend ${trendClass}">${trendArrow}</span>
                    <span>${cfg.meta(v)}</span>
                </div>
            </div>
        `;
    }).join('');

    // Update badges
    const critCount = data.critical_alerts?.value || 0;
    document.getElementById('notif-badge').textContent = critCount;
    document.getElementById('critical-count').textContent = `${critCount} Critical`;
    document.getElementById('drainage-badge').textContent = critCount;

    initLucideIcons();
}

// ─── CHART RENDERING ────────────────────────────────────────────────
function renderChart(canvasId, data, label, borderColor, bgColor, dangerLine) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    // Take last 48 points for readability
    const sliced = data.slice(-48);
    const labels = sliced.map(d => d.label);
    const values = sliced.map(d => d.value);

    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    const datasets = [{
        label: label,
        data: values,
        borderColor: borderColor,
        backgroundColor: bgColor,
        fill: true,
        tension: 0.4,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBackgroundColor: borderColor,
        pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 2,
    }];

    // Add danger threshold line if specified
    if (dangerLine) {
        datasets.push({
            label: 'Danger Threshold',
            data: Array(values.length).fill(dangerLine),
            borderColor: '#ef4444',
            borderWidth: 1.5,
            borderDash: [6, 4],
            fill: false,
            pointRadius: 0,
            pointHoverRadius: 0,
        });
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#111a2e',
                    borderColor: '#1e2a42',
                    borderWidth: 1,
                    titleColor: '#e8ecf4',
                    bodyColor: '#8892a8',
                    titleFont: { family: 'Inter', weight: '600' },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false,
                }
            },
            scales: {
                x: {
                    grid: { color: '#1e2a4230', drawBorder: false },
                    ticks: { color: '#5a6478', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 10, maxRotation: 0 },
                    border: { display: false },
                },
                y: {
                    grid: { color: '#1e2a4230', drawBorder: false },
                    ticks: { color: '#5a6478', font: { family: 'JetBrains Mono', size: 10 } },
                    border: { display: false },
                }
            },
            animation: { duration: 800, easing: 'easeOutQuart' },
        }
    });
}

function updateChartData(canvasId, data) {
    if (!charts[canvasId]) return;
    const sliced = data.slice(-48);
    charts[canvasId].data.labels = sliced.map(d => d.label);
    charts[canvasId].data.datasets[0].data = sliced.map(d => d.value);
    charts[canvasId].update('none'); // No animation on update for smoothness
}

// ─── MAP ────────────────────────────────────────────────────────────
function initMap() {
    const mapContainer = document.getElementById('city-map');
    if (!mapContainer) return;

    cityMap = L.map('city-map', {
        zoomControl: false,
        attributionControl: false,
    }).setView([13.04, 80.23], 11);

    // Dark map tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        maxZoom: 19,
    }).addTo(cityMap);

    L.control.zoom({ position: 'topright' }).addTo(cityMap);

    // Map layer filter buttons
    document.querySelectorAll('.map-layer-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.map-layer-btn').forEach(b => {
                b.style.background = 'var(--bg-secondary)';
                b.style.color = 'var(--text-muted)';
                b.style.borderColor = 'var(--border)';
                b.classList.remove('active');
            });
            btn.style.background = 'var(--cyan-dim)';
            btn.style.color = 'var(--cyan)';
            btn.style.borderColor = 'var(--cyan)';
            btn.classList.add('active');
        });
    });
}

function updateMapMarkers(nodes) {
    if (!cityMap) return;

    // Clear existing markers
    mapMarkers.forEach(m => cityMap.removeLayer(m));
    mapMarkers = [];

    nodes.forEach(node => {
        let markerClass = 'marker-online';
        if (node.status === 'offline') markerClass = 'marker-offline';
        else if (node.urgency === 'critical' || node.risk_score > 75) markerClass = 'marker-critical';
        else if (node.urgency === 'high' || node.risk_score > 55) markerClass = 'marker-warning';

        const size = markerClass === 'marker-critical' ? 16 : 12;

        const icon = L.divIcon({
            className: `custom-marker ${markerClass}`,
            iconSize: [size, size],
            iconAnchor: [size / 2, size / 2],
        });

        const marker = L.marker([node.lat, node.lng], { icon })
            .bindPopup(`
                <div class="popup-node-id">${node.id}</div>
                <div class="popup-node-name">${node.name}</div>
                <div class="popup-stat"><span class="popup-stat-label">Zone</span><span class="popup-stat-value">${node.zone}</span></div>
                <div class="popup-stat"><span class="popup-stat-label">Status</span><span class="popup-stat-value" style="color:${getStatusColor(node.status)}">${node.status.toUpperCase()}</span></div>
                <div class="popup-stat"><span class="popup-stat-label">Risk Score</span><span class="popup-stat-value" style="color:${getRiskColor(node.risk_score)}">${node.risk_score}/100</span></div>
                <div class="popup-stat"><span class="popup-stat-label">Water Level</span><span class="popup-stat-value">${node.last_reading}m</span></div>
                <div class="popup-stat"><span class="popup-stat-label">Flow Rate</span><span class="popup-stat-value">${node.flow_rate} m³/s</span></div>
                <div class="popup-stat"><span class="popup-stat-label">Battery</span><span class="popup-stat-value">${node.battery}%</span></div>
                <div class="popup-stat"><span class="popup-stat-label">Updated</span><span class="popup-stat-value">${node.last_updated}</span></div>
            `)
            .addTo(cityMap);

        mapMarkers.push(marker);
    });
}

function getStatusColor(status) {
    const colors = { online: '#10b981', warning: '#f59e0b', offline: '#ef4444', maintenance: '#a855f7' };
    return colors[status] || '#5a6478';
}

function getRiskColor(score) {
    if (score > 75) return '#ef4444';
    if (score > 55) return '#f59e0b';
    if (score > 35) return '#3b82f6';
    return '#10b981';
}

// ─── RISK GRID ──────────────────────────────────────────────────────
function renderRiskGrid(zones) {
    const html = zones.map(zone => {
        const trendIcon = zone.trend === 'rising' ? '▲' : zone.trend === 'falling' ? '▼' : '●';
        const trendColor = zone.trend === 'rising' ? 'var(--red)' : zone.trend === 'falling' ? 'var(--emerald)' : 'var(--text-muted)';
        return `
            <div class="risk-zone ${zone.risk_level}">
                <div class="risk-zone-glow ${zone.risk_level}"></div>
                <div class="risk-zone-name">${zone.name}</div>
                <div class="risk-zone-score ${zone.risk_level}">${zone.risk_score}</div>
                <span class="urgency-badge ${zone.risk_level}">${zone.risk_level}</span>
                <div class="risk-zone-meta">
                    <span>🔌 ${zone.active_sensors} sensors</span>
                    <span>⚠️ ${zone.alerts_count} alerts</span>
                    <span style="color:${trendColor}">${trendIcon} ${zone.trend}</span>
                </div>
            </div>
        `;
    }).join('');

    const el = document.getElementById('risk-grid');
    if (el) el.innerHTML = html;
    const analyticsEl = document.getElementById('analytics-risk-grid');
    if (analyticsEl) analyticsEl.innerHTML = html;
}

// ─── ALERTS FEED ────────────────────────────────────────────────────
function renderAlerts(alerts) {
    const html = alerts.slice(0, 8).map(alert => `
        <div class="alert-item ${alert.severity}" title="Click for details">
            <span class="alert-severity ${alert.severity}">${alert.severity}</span>
            <div class="alert-content">
                <div class="alert-title">${alert.title}</div>
                <div class="alert-message">${alert.message}</div>
                <div class="alert-prediction">${alert.prediction}</div>
            </div>
            <span class="alert-time">${alert.timestamp.split(' ')[1]}</span>
        </div>
    `).join('');

    const el = document.getElementById('alerts-feed');
    if (el) el.innerHTML = html;
    // Also populate drainage page alerts & AI insights page alerts
    const drEl = document.getElementById('drainage-alerts-feed');
    if (drEl) drEl.innerHTML = html;
    const aiEl = document.getElementById('ai-page-alerts');
    if (aiEl) aiEl.innerHTML = html;
}

// ─── DRAINAGE TABLE ─────────────────────────────────────────────────
function renderDrainageTable(nodes) {
    const tbody = document.getElementById('drainage-tbody');
    document.getElementById('node-count').textContent = `${nodes.length} Nodes`;

    tbody.innerHTML = nodes.map(node => {
        const riskColor = getRiskColor(node.risk_score);
        const riskWidth = node.risk_score;
        return `
            <tr>
                <td><span style="font-family:var(--font-mono);color:var(--cyan);font-weight:600">${node.id}</span></td>
                <td>${node.name}</td>
                <td><span style="color:var(--text-muted)">${node.zone}</span></td>
                <td><span class="status-chip ${node.status}">${node.status}</span></td>
                <td style="font-family:var(--font-mono)">${node.last_reading}</td>
                <td style="font-family:var(--font-mono)">${node.flow_rate}</td>
                <td>
                    <div class="risk-bar"><div class="risk-bar-fill" style="width:${riskWidth}%;background:${riskColor}"></div></div>
                    <span style="font-family:var(--font-mono);font-weight:600;color:${riskColor}">${node.risk_score}</span>
                </td>
                <td><span class="urgency-badge ${node.urgency}">${node.urgency}</span></td>
                <td style="font-family:var(--font-mono);color:var(--text-muted)">${node.last_updated}</td>
            </tr>
        `;
    }).join('');
}

// ─── REPAIR RECOMMENDATIONS ────────────────────────────────────────
function renderRepairs(recommendations) {
    const html = recommendations.map(rec => `
        <div class="repair-card ${rec.priority}">
            <div class="repair-header">
                <span class="repair-priority ${rec.priority}">${rec.priority}</span>
                <span style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted)">Confidence: ${rec.confidence}%</span>
            </div>
            <div class="repair-title">${rec.title}</div>
            <div class="repair-location">📍 ${rec.location}</div>
            <div class="repair-issue">${rec.issue}</div>
            <div class="repair-action">💡 ${rec.action}</div>
            <div class="repair-meta">
                <div class="repair-meta-item">⏱️ Failure Window: <span>${rec.failure_window}</span></div>
            </div>
            <div style="margin-top:8px;font-size:0.72rem;color:var(--text-muted)">🎯 Impact: ${rec.impact}</div>
        </div>
    `).join('');

    const el = document.getElementById('repair-list');
    if (el) el.innerHTML = html;
    const drEl = document.getElementById('drainage-repair-list');
    if (drEl) drEl.innerHTML = html;
}

// ─── AI INSIGHTS ────────────────────────────────────────────────────
function renderInsights(insights) {
    const html = insights.map(ins => `
        <div class="insight-card">
            <span class="insight-category ${ins.category.toLowerCase()}">${ins.category}</span>
            <div class="insight-title">${ins.title}</div>
            <div class="insight-desc">${ins.description}</div>
            <div class="insight-action">➡️ ${ins.action}</div>
            <div class="insight-footer">
                <span style="font-family:var(--font-mono);font-size:0.68rem">🤖 ${ins.model}</span>
                <div class="confidence-bar">
                    <span>Confidence:</span>
                    <div class="bar"><div class="bar-fill" style="width:${ins.confidence}%"></div></div>
                    <span style="font-family:var(--font-mono);font-weight:600">${ins.confidence}%</span>
                </div>
            </div>
        </div>
    `).join('');

    const el = document.getElementById('insights-list');
    if (el) el.innerHTML = html;
    const aiEl = document.getElementById('ai-page-insights');
    if (aiEl) aiEl.innerHTML = html;
}

// ─── SYSTEM STATUS ──────────────────────────────────────────────────
function updateSystemStatus(status) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    if (status.status === 'operational') {
        dot.className = 'status-dot green';
        text.textContent = 'All Systems Operational';
    } else {
        dot.className = 'status-dot amber';
        text.textContent = 'Partial Degradation';
    }
}

// ─── ANALYTICS PAGE CHARTS ──────────────────────────────────────────
let analyticsInitialized = false;
async function initAnalyticsCharts() {
    if (analyticsInitialized) return;
    analyticsInitialized = true;

    const [rainfall, water, soil] = await Promise.all([
        api.getSensorData('rainfall'),
        api.getSensorData('water_level'),
        api.getSensorData('soil_moisture'),
    ]);

    if (rainfall) renderChart('analytics-rainfall-chart', rainfall, 'Rainfall', '#06d6a0', 'rgba(6,214,160,0.1)');
    if (water) renderChart('analytics-water-chart', water, 'Water Level', '#3b82f6', 'rgba(59,130,246,0.1)', 4.2);
    if (soil) renderChart('analytics-soil-chart', soil, 'Soil Moisture', '#a855f7', 'rgba(168,85,247,0.1)');

    initLucideIcons();
}


// ═══════════════════════════════════════════════════════════════════
// NEW PAGES: STRESS SCORE, ROAD HEALTH, BRIDGE HEALTH
// ═══════════════════════════════════════════════════════════════════

// ─── STRESS SCORE PAGE ──────────────────────────────────────────────
let stressPageLoaded = false;
async function loadStressScorePage() {
    if (stressPageLoaded) return;
    stressPageLoaded = true;

    const [scoreData, historyData] = await Promise.all([
        api.getStressScore(),
        api.getStressHistory(),
    ]);

    if (scoreData) renderStressComparison(scoreData);
    if (historyData) {
        renderStressHistoryChart(historyData);
        renderStressSensorCharts(historyData);
    }

    initLucideIcons();
}

function renderStressComparison(data) {
    const container = document.getElementById('stress-comparison-row');
    if (!container) return;

    // Update model info banner
    if (data.model_info) {
        const nameEl = document.getElementById('stress-model-name');
        const r2El = document.getElementById('stress-model-r2');
        if (nameEl) nameEl.textContent = `${data.model_info.name} ${data.model_info.version}`;
        if (r2El) {
            r2El.querySelector('.r2-value').textContent = data.model_info.r2_score.toFixed(2);
        }
    }

    function getScoreLevel(score) {
        if (score > 70) return { level: 'CRITICAL', color: 'var(--red)', bg: 'var(--red-dim)' };
        if (score > 45) return { level: 'MODERATE', color: 'var(--amber)', bg: 'var(--amber-dim)' };
        return { level: 'LOW', color: 'var(--emerald)', bg: 'rgba(16,185,129,0.15)' };
    }

    const days = [
        { label: 'Latest Reading', key: 'today', primary: true },
        { label: 'Previous Day', key: 'yesterday', primary: false },
        { label: 'Day Before', key: 'day_before', primary: false },
    ];

    container.innerHTML = days.map((day, i) => {
        const d = data[day.key];
        const sl = getScoreLevel(d.score);
        return `
            <div class="stress-card ${day.primary ? 'primary' : ''} animate-in" style="animation-delay:${i * 0.1}s">
                <div class="stress-card-glow" style="background:${sl.color}"></div>
                <div class="stress-card-label">${day.label}</div>
                <div class="stress-card-date">${d.date}</div>
                <div class="stress-card-score" style="color:${sl.color}">${d.score}</div>
                <span class="stress-card-level" style="color:${sl.color};background:${sl.bg}">${sl.level}</span>
                <div class="stress-card-sensors">
                    <div class="sensor-reading"><span class="sensor-label">Rainfall</span><span class="sensor-value">${d.rainfall} mm</span></div>
                    <div class="sensor-reading"><span class="sensor-label">Soil Moisture</span><span class="sensor-value">${d.soil_moisture}%</span></div>
                    <div class="sensor-reading"><span class="sensor-label">Drain Level</span><span class="sensor-value">${d.drain_level} m</span></div>
                </div>
            </div>
        `;
    }).join('');

    // Add delta indicator to primary card
    const deltaHtml = `
        <div class="stress-delta ${data.trend}" style="margin-top:8px">
            <span class="delta-arrow">${data.delta > 0 ? '▲' : data.delta < 0 ? '▼' : '●'}</span>
            <span class="delta-value">${Math.abs(data.delta).toFixed(1)}</span>
            <span class="delta-label">${data.trend}</span>
        </div>
    `;
    const primaryCard = container.querySelector('.stress-card.primary');
    if (primaryCard) {
        primaryCard.insertAdjacentHTML('beforeend', deltaHtml);
    }
}

function renderStressHistoryChart(data) {
    const ctx = document.getElementById('stress-history-chart');
    if (!ctx) return;

    if (charts['stress-history-chart']) charts['stress-history-chart'].destroy();

    const labels = data.map(d => d.label);
    const scores = data.map(d => d.score);

    // Color segments based on score level
    const pointColors = scores.map(s => s > 70 ? '#ef4444' : s > 45 ? '#f59e0b' : '#10b981');

    charts['stress-history-chart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Stress Score',
                    data: scores,
                    borderColor: '#06d6a0',
                    backgroundColor: 'rgba(6,214,160,0.08)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 2.5,
                    pointRadius: 4,
                    pointBackgroundColor: pointColors,
                    pointBorderColor: pointColors,
                    pointHoverRadius: 7,
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                },
                {
                    label: 'High Risk Threshold',
                    data: Array(scores.length).fill(70),
                    borderColor: '#ef444480',
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    fill: false,
                    pointRadius: 0,
                },
                {
                    label: 'Moderate Threshold',
                    data: Array(scores.length).fill(45),
                    borderColor: '#f59e0b80',
                    borderWidth: 1.5,
                    borderDash: [6, 4],
                    fill: false,
                    pointRadius: 0,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#8892a8',
                        font: { family: 'Inter', size: 11 },
                        usePointStyle: true,
                        padding: 20,
                    }
                },
                tooltip: {
                    backgroundColor: '#111a2e',
                    borderColor: '#1e2a42',
                    borderWidth: 1,
                    titleColor: '#e8ecf4',
                    bodyColor: '#8892a8',
                    titleFont: { family: 'Inter', weight: '600' },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                },
            },
            scales: {
                x: {
                    grid: { color: '#1e2a4230', drawBorder: false },
                    ticks: { color: '#5a6478', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 15, maxRotation: 45 },
                    border: { display: false },
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: '#1e2a4230', drawBorder: false },
                    ticks: { color: '#5a6478', font: { family: 'JetBrains Mono', size: 10 } },
                    border: { display: false },
                },
            },
            animation: { duration: 1000, easing: 'easeOutQuart' },
        },
    });
}

function renderStressSensorCharts(data) {
    // Rainfall chart on stress page
    const rainfallData = data.map(d => ({ label: d.label, value: d.rainfall }));
    renderChart('stress-rainfall-chart', rainfallData, 'Rainfall', '#3b82f6', 'rgba(59,130,246,0.1)');

    // Soil moisture chart on stress page
    const soilData = data.map(d => ({ label: d.label, value: d.soil_moisture }));
    renderChart('stress-soil-chart', soilData, 'Soil Moisture', '#a855f7', 'rgba(168,85,247,0.1)');
}


// ─── ROAD HEALTH PAGE ───────────────────────────────────────────────
let roadPageLoaded = false;
async function loadRoadHealthPage() {
    if (roadPageLoaded) return;
    roadPageLoaded = true;

    const data = await api.getRoadHealth();
    if (!data) return;

    renderRoadKPIs(data.summary);
    renderRoadHealthChart(data.history);
    renderRoadTable(data.segments);

    initLucideIcons();
}

function renderRoadKPIs(summary) {
    const container = document.getElementById('road-kpi-row');
    if (!container) return;

    const healthColor = summary.overall_health > 70 ? 'emerald' : summary.overall_health > 50 ? 'amber' : 'red';

    const kpis = [
        { label: 'Overall Health', value: `${summary.overall_health}%`, icon: 'heart-pulse', color: healthColor },
        { label: 'Total Segments', value: summary.total_segments, icon: 'map', color: 'cyan' },
        { label: 'Total Potholes', value: summary.total_potholes, icon: 'alert-circle', color: 'amber' },
        { label: 'Critical Segments', value: summary.critical_segments, icon: 'alert-triangle', color: 'red' },
    ];

    container.innerHTML = kpis.map((kpi, i) => `
        <div class="card kpi-card animate-in" style="animation-delay:${i * 0.05}s">
            <div class="kpi-glow ${kpi.color}"></div>
            <div class="kpi-icon ${kpi.color}">
                <i data-lucide="${kpi.icon}" style="width:22px;height:22px"></i>
            </div>
            <div class="kpi-label">${kpi.label}</div>
            <div class="kpi-value" style="color:var(--${kpi.color})">${kpi.value}</div>
            <div class="kpi-meta">
                <span class="kpi-trend ${summary.trend === 'declining' ? 'up' : summary.trend === 'improving' ? 'down' : 'stable'}">${summary.trend === 'declining' ? '▲' : summary.trend === 'improving' ? '▼' : '●'}</span>
                <span>Trend: ${summary.trend}</span>
            </div>
        </div>
    `).join('');
}

function renderRoadHealthChart(history) {
    const ctx = document.getElementById('road-health-chart');
    if (!ctx) return;

    if (charts['road-health-chart']) charts['road-health-chart'].destroy();

    charts['road-health-chart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: history.map(d => d.label),
            datasets: [{
                label: 'Road Health Index',
                data: history.map(d => d.score),
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245,158,11,0.08)',
                fill: true,
                tension: 0.4,
                borderWidth: 2.5,
                pointRadius: 3,
                pointBackgroundColor: '#f59e0b',
                pointHoverRadius: 6,
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#111a2e',
                    borderColor: '#1e2a42',
                    borderWidth: 1,
                    titleColor: '#e8ecf4',
                    bodyColor: '#8892a8',
                    titleFont: { family: 'Inter', weight: '600' },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false,
                },
            },
            scales: {
                x: {
                    grid: { color: '#1e2a4230', drawBorder: false },
                    ticks: { color: '#5a6478', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 10, maxRotation: 0 },
                    border: { display: false },
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: '#1e2a4230', drawBorder: false },
                    ticks: { color: '#5a6478', font: { family: 'JetBrains Mono', size: 10 } },
                    border: { display: false },
                },
            },
            animation: { duration: 800, easing: 'easeOutQuart' },
        },
    });
}

function renderRoadTable(segments) {
    const tbody = document.getElementById('road-tbody');
    const countEl = document.getElementById('road-segment-count');
    if (countEl) countEl.textContent = `${segments.length} Segments`;

    tbody.innerHTML = segments.map(seg => {
        const healthColor = getRiskColor(100 - seg.health_score); // Invert for health
        return `
            <tr>
                <td><span style="font-family:var(--font-mono);color:var(--amber);font-weight:600">${seg.id}</span></td>
                <td>${seg.name}</td>
                <td><span style="color:var(--text-muted)">${seg.zone}</span></td>
                <td style="font-family:var(--font-mono)">${seg.length_km}</td>
                <td>
                    <div class="risk-bar"><div class="risk-bar-fill" style="width:${seg.health_score}%;background:${seg.health_score > 70 ? '#10b981' : seg.health_score > 50 ? '#f59e0b' : '#ef4444'}"></div></div>
                    <span style="font-family:var(--font-mono);font-weight:600;color:${seg.health_score > 70 ? '#10b981' : seg.health_score > 50 ? '#f59e0b' : '#ef4444'}">${seg.health_score}</span>
                </td>
                <td><span class="status-chip ${seg.condition}">${seg.condition}</span></td>
                <td style="font-family:var(--font-mono);color:${seg.pothole_count > 20 ? 'var(--red)' : seg.pothole_count > 10 ? 'var(--amber)' : 'var(--text-primary)'};font-weight:600">${seg.pothole_count}</td>
                <td style="font-family:var(--font-mono)">${seg.surface_index}</td>
                <td><span class="urgency-badge ${seg.traffic_load === 'very high' ? 'critical' : seg.traffic_load === 'high' ? 'high' : seg.traffic_load === 'medium' ? 'medium' : 'low'}">${seg.traffic_load}</span></td>
                <td style="font-family:var(--font-mono);color:var(--text-muted)">${seg.last_inspection}</td>
            </tr>
        `;
    }).join('');
}


// ─── BRIDGE HEALTH PAGE ─────────────────────────────────────────────
let bridgePageLoaded = false;
async function loadBridgeHealthPage() {
    if (bridgePageLoaded) return;
    bridgePageLoaded = true;

    const data = await api.getBridgeHealth();
    if (!data) return;

    renderBridgeKPIs(data.summary);
    renderBridgeHealthChart(data.history);
    renderBridgeTable(data.bridges);

    initLucideIcons();
}

function renderBridgeKPIs(summary) {
    const container = document.getElementById('bridge-kpi-row');
    if (!container) return;

    const healthColor = summary.overall_health > 70 ? 'emerald' : summary.overall_health > 50 ? 'amber' : 'red';

    const kpis = [
        { label: 'Structural Health', value: `${summary.overall_health}%`, icon: 'shield-check', color: healthColor },
        { label: 'Total Bridges', value: summary.total_bridges, icon: 'bridge', color: 'cyan' },
        { label: 'Bridges at Risk', value: summary.bridges_at_risk, icon: 'alert-triangle', color: 'red' },
        { label: 'Active Monitors', value: summary.active_monitors, icon: 'radio-tower', color: 'purple' },
    ];

    container.innerHTML = kpis.map((kpi, i) => `
        <div class="card kpi-card animate-in" style="animation-delay:${i * 0.05}s">
            <div class="kpi-glow ${kpi.color}"></div>
            <div class="kpi-icon ${kpi.color}">
                <i data-lucide="${kpi.icon}" style="width:22px;height:22px"></i>
            </div>
            <div class="kpi-label">${kpi.label}</div>
            <div class="kpi-value" style="color:var(--${kpi.color})">${kpi.value}</div>
            <div class="kpi-meta">
                <span class="kpi-trend ${summary.trend === 'declining' ? 'up' : summary.trend === 'improving' ? 'down' : 'stable'}">${summary.trend === 'declining' ? '▲' : summary.trend === 'improving' ? '▼' : '●'}</span>
                <span>Trend: ${summary.trend}</span>
            </div>
        </div>
    `).join('');
}

function renderBridgeHealthChart(history) {
    const ctx = document.getElementById('bridge-health-chart');
    if (!ctx) return;

    if (charts['bridge-health-chart']) charts['bridge-health-chart'].destroy();

    charts['bridge-health-chart'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: history.map(d => d.label),
            datasets: [{
                label: 'Structural Health Index',
                data: history.map(d => d.score),
                borderColor: '#a855f7',
                backgroundColor: 'rgba(168,85,247,0.08)',
                fill: true,
                tension: 0.4,
                borderWidth: 2.5,
                pointRadius: 3,
                pointBackgroundColor: '#a855f7',
                pointHoverRadius: 6,
                pointHoverBorderColor: '#fff',
                pointHoverBorderWidth: 2,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#111a2e',
                    borderColor: '#1e2a42',
                    borderWidth: 1,
                    titleColor: '#e8ecf4',
                    bodyColor: '#8892a8',
                    titleFont: { family: 'Inter', weight: '600' },
                    bodyFont: { family: 'JetBrains Mono', size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false,
                },
            },
            scales: {
                x: {
                    grid: { color: '#1e2a4230', drawBorder: false },
                    ticks: { color: '#5a6478', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 10, maxRotation: 0 },
                    border: { display: false },
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: '#1e2a4230', drawBorder: false },
                    ticks: { color: '#5a6478', font: { family: 'JetBrains Mono', size: 10 } },
                    border: { display: false },
                },
            },
            animation: { duration: 800, easing: 'easeOutQuart' },
        },
    });
}

function renderBridgeTable(bridges) {
    const tbody = document.getElementById('bridge-tbody');
    const countEl = document.getElementById('bridge-count');
    if (countEl) countEl.textContent = `${bridges.length} Bridges`;

    tbody.innerHTML = bridges.map(bridge => {
        const statusColors = {
            healthy: 'var(--emerald)',
            monitor: 'var(--blue)',
            warning: 'var(--amber)',
            critical: 'var(--red)',
        };
        return `
            <tr>
                <td><span style="font-family:var(--font-mono);color:var(--purple);font-weight:600">${bridge.id}</span></td>
                <td>${bridge.name}</td>
                <td><span style="color:var(--text-muted)">${bridge.type}</span></td>
                <td><span style="color:var(--text-muted)">${bridge.zone}</span></td>
                <td>
                    <div class="risk-bar"><div class="risk-bar-fill" style="width:${bridge.structural_health}%;background:${bridge.structural_health > 70 ? '#10b981' : bridge.structural_health > 50 ? '#f59e0b' : '#ef4444'}"></div></div>
                    <span style="font-family:var(--font-mono);font-weight:600;color:${bridge.structural_health > 70 ? '#10b981' : bridge.structural_health > 50 ? '#f59e0b' : '#ef4444'}">${bridge.structural_health}</span>
                </td>
                <td><span class="status-chip ${bridge.status}" style="color:${statusColors[bridge.status]}">${bridge.status}</span></td>
                <td style="font-family:var(--font-mono);color:${bridge.vibration_score > 3 ? 'var(--red)' : bridge.vibration_score > 2 ? 'var(--amber)' : 'var(--text-primary)'}">${bridge.vibration_score}</td>
                <td style="font-family:var(--font-mono);color:${bridge.load_factor > 0.9 ? 'var(--red)' : bridge.load_factor > 0.7 ? 'var(--amber)' : 'var(--text-primary)'}">${bridge.load_factor}</td>
                <td style="font-family:var(--font-mono);color:${bridge.crack_index > 5 ? 'var(--red)' : bridge.crack_index > 3 ? 'var(--amber)' : 'var(--text-primary)'}">${bridge.crack_index}</td>
                <td style="font-family:var(--font-mono);color:var(--text-muted)">${bridge.next_maintenance}</td>
            </tr>
        `;
    }).join('');
}
