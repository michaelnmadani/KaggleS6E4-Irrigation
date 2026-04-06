import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis, LineChart, Line
} from 'recharts';
import {
  Droplets, Sun, Cloud, Leaf, BarChart3, CheckCircle2,
  AlertTriangle, XCircle, ChevronDown, ChevronUp, Database,
  Cpu, FlaskConical, ClipboardCheck
} from 'lucide-react';

// ─── Fallback / sample data ────────────────────────────────────────────
// This data is replaced by outputs/results.json when the pipeline runs.
const SAMPLE_RESULTS = {
  competition: {
    name: "Predicting Irrigation Need",
    slug: "playground-series-s6e4",
    metric: "balanced_accuracy",
    target: "Irrigation_Need",
    classes: ["Low", "Medium", "High"],
  },
  dataset: {
    train_shape: [0, 0],
    test_shape: [0, 0],
    train_columns: [],
    dtypes: {},
  },
  eda: {
    class_distribution: { counts: { Low: 0, Medium: 0, High: 0 }, percentages: { Low: 33, Medium: 34, High: 33 } },
    missing_values: { train: {}, test: {}, train_total: 0, test_total: 0 },
    numeric_stats: {},
    categorical_stats: {},
    correlations: {},
    sample_data: { columns: [], rows: [] },
    feature_categories: {
      soil: ["Soil_Type", "Soil_pH", "Soil_Moisture", "Soil_Organic_Carbon", "Soil_Electrical_Conductivity"],
      weather: ["Temperature", "Humidity", "Rainfall", "Sunlight_Hours", "Wind_Speed"],
      crop: ["Crop_Type", "Growth_Stage", "Season", "Irrigation_Method", "Water_Source"],
      field: ["Field_Area", "Mulching", "Prev_Irrigation_Amount", "Region"],
    },
  },
  pipeline: { steps: [] },
  models: {},
  best_model: "",
  best_score: 0,
  review: {
    verdict: "PENDING",
    issues: [],
    warnings: [],
    approvals: [],
    model_scores: {},
    prediction_distribution: {},
    training_distribution: {},
  },
  elapsed_seconds: 0,
};

// Load pipeline results (empty object if pipeline hasn't run yet)
import pipelineResults from '../outputs/results.json';
const DATA = { ...SAMPLE_RESULTS, ...pipelineResults };

const COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'];
const CLASS_COLORS = { Low: '#22c55e', Medium: '#f59e0b', High: '#ef4444' };

// ─── Utility Components ────────────────────────────────────────────────

function SectionHeader({ icon: Icon, title, subtitle }) {
  return (
    <div className="mb-6">
      <div className="flex items-center gap-3 mb-1">
        <Icon className="w-7 h-7 text-brand-600" />
        <h2 className="text-2xl font-bold text-gray-900">{title}</h2>
      </div>
      {subtitle && <p className="text-gray-500 ml-10">{subtitle}</p>}
      <div className="h-0.5 bg-gradient-to-r from-brand-500 to-transparent mt-3" />
    </div>
  );
}

function StatCard({ label, value, sub, color = 'brand' }) {
  const colorMap = {
    brand: 'bg-brand-50 border-brand-200 text-brand-700',
    blue: 'bg-blue-50 border-blue-200 text-blue-700',
    amber: 'bg-amber-50 border-amber-200 text-amber-700',
    red: 'bg-red-50 border-red-200 text-red-700',
  };
  return (
    <div className={`rounded-xl border p-4 ${colorMap[color]}`}>
      <div className="text-sm font-medium opacity-75">{label}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
      {sub && <div className="text-xs mt-1 opacity-60">{sub}</div>}
    </div>
  );
}

function CodeBlock({ code, title }) {
  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
      {title && <div className="px-4 py-2 bg-gray-800 text-gray-400 text-xs font-mono">{title}</div>}
      <pre className="p-4 text-sm text-green-400 font-mono overflow-x-auto">{code}</pre>
    </div>
  );
}

function Badge({ text, variant = 'default' }) {
  const styles = {
    default: 'bg-gray-100 text-gray-700',
    success: 'bg-green-100 text-green-700',
    warning: 'bg-amber-100 text-amber-700',
    error: 'bg-red-100 text-red-700',
  };
  return <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${styles[variant]}`}>{text}</span>;
}

// ─── Section 1: Introduction ───────────────────────────────────────────

function IntroductionSection() {
  const d = DATA;
  const cats = d.eda?.feature_categories || {};
  const catIcons = { soil: Droplets, weather: Cloud, crop: Leaf, field: BarChart3 };
  const catLabels = { soil: 'Soil Properties', weather: 'Weather Data', crop: 'Crop Info', field: 'Field Characteristics' };

  const classDist = d.eda?.class_distribution?.counts || {};
  const classData = Object.entries(classDist).map(([name, value]) => ({
    name, value, fill: CLASS_COLORS[name] || '#888'
  }));

  return (
    <section className="mb-12">
      <SectionHeader icon={Database} title="Introduction" subtitle="Dataset overview and competition context" />

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <StatCard label="Competition" value={d.competition?.name || 'S6E4'} sub="Playground Series" />
        <StatCard label="Train Rows" value={(d.dataset?.train_shape?.[0] || 0).toLocaleString()} sub={`${d.dataset?.train_shape?.[1] || 0} columns`} color="blue" />
        <StatCard label="Test Rows" value={(d.dataset?.test_shape?.[0] || 0).toLocaleString()} color="blue" />
        <StatCard label="Metric" value="Balanced Accuracy" sub="Multi-class" color="amber" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Feature Categories (20 features)</h3>
          <div className="space-y-3">
            {Object.entries(cats).map(([key, features]) => {
              const Icon = catIcons[key] || BarChart3;
              return (
                <div key={key} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                  <Icon className="w-5 h-5 text-brand-600 mt-0.5 shrink-0" />
                  <div>
                    <div className="font-medium text-gray-800">{catLabels[key] || key}</div>
                    <div className="text-sm text-gray-500">{features.join(', ')}</div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Target: Irrigation Need</h3>
          <div className="flex items-center gap-3 mb-4">
            {['Low', 'Medium', 'High'].map(label => (
              <Badge key={label} text={label} variant={label === 'Low' ? 'success' : label === 'Medium' ? 'warning' : 'error'} />
            ))}
          </div>
          {classData.length > 0 && (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={classData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" name="Count">
                  {classData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </section>
  );
}

// ─── Section 2: Data Cleaning & Feature Engineering ────────────────────

function DataCleaningSection() {
  const { eda, pipeline } = DATA;
  const steps = pipeline?.steps || [];
  const sample = eda?.sample_data || { columns: [], rows: [] };
  const missing = eda?.missing_values || {};
  const numStats = eda?.numeric_stats || {};

  const [showAllStats, setShowAllStats] = useState(false);
  const statEntries = Object.entries(numStats);
  const visibleStats = showAllStats ? statEntries : statEntries.slice(0, 6);

  return (
    <section className="mb-12">
      <SectionHeader icon={FlaskConical} title="Data Cleaning & Feature Engineering" subtitle="Preprocessing pipeline and feature analysis" />

      {/* Raw data sample */}
      {sample.columns.length > 0 && (
        <div className="mb-6">
          <h3 className="font-semibold text-gray-800 mb-2">Raw Data Sample</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm border border-gray-200 rounded-lg overflow-hidden">
              <thead className="bg-gray-100">
                <tr>
                  {sample.columns.map(col => (
                    <th key={col} className="px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sample.rows.map((row, i) => (
                  <tr key={i} className={i % 2 ? 'bg-gray-50' : ''}>
                    {row.map((cell, j) => (
                      <td key={j} className="px-3 py-1.5 whitespace-nowrap text-gray-700">{cell != null ? String(cell) : '—'}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Missing values */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-800 mb-2">Missing Values</h3>
          {missing.train_total === 0 && missing.test_total === 0 ? (
            <div className="flex items-center gap-2 text-green-600">
              <CheckCircle2 className="w-5 h-5" />
              <span>No missing values in train or test</span>
            </div>
          ) : (
            <div className="space-y-1 text-sm">
              <div>Train: {missing.train_total} missing</div>
              <div>Test: {missing.test_total} missing</div>
              {Object.entries(missing.train || {}).map(([col, count]) => (
                <div key={col} className="text-gray-500">  {col}: {count}</div>
              ))}
            </div>
          )}
        </div>

        {/* Numeric stats summary */}
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-800 mb-2">Numeric Feature Stats</h3>
          {visibleStats.length > 0 ? (
            <div className="space-y-1 text-sm">
              {visibleStats.map(([feat, stats]) => (
                <div key={feat} className="flex justify-between">
                  <span className="text-gray-600">{feat}</span>
                  <span className="font-mono text-gray-800">
                    {stats.mean !== undefined ? `mean=${Number(stats.mean).toFixed(2)}` : ''} {stats.std !== undefined ? `std=${Number(stats.std).toFixed(2)}` : ''}
                  </span>
                </div>
              ))}
              {statEntries.length > 6 && (
                <button onClick={() => setShowAllStats(!showAllStats)} className="text-brand-600 text-xs flex items-center gap-1 mt-2">
                  {showAllStats ? <><ChevronUp className="w-3 h-3" /> Show less</> : <><ChevronDown className="w-3 h-3" /> Show all ({statEntries.length})</>}
                </button>
              )}
            </div>
          ) : (
            <div className="text-gray-400 text-sm">Run pipeline to see stats</div>
          )}
        </div>
      </div>

      {/* Pipeline steps */}
      {steps.length > 0 && (
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Preprocessing Pipeline</h3>
          <div className="space-y-3">
            {steps.map((step) => (
              <div key={step.step} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-1">
                  <span className="bg-brand-100 text-brand-700 text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center">{step.step}</span>
                  <span className="font-medium text-gray-800">{step.name}</span>
                </div>
                <p className="text-sm text-gray-500 ml-8 mb-2">{step.description}</p>
                {step.code && <div className="ml-8"><CodeBlock code={step.code} /></div>}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

// ─── Section 3: Model Training ─────────────────────────────────────────

function ModelTrainingSection() {
  const models = DATA.models || {};
  const modelEntries = Object.entries(models);

  const modelColors = {
    xgboost: '#ef4444',
    lightgbm: '#22c55e',
    random_forest: '#3b82f6',
    logistic_regression: '#f59e0b',
    ensemble: '#8b5cf6',
  };

  const modelLabels = {
    xgboost: 'XGBoost',
    lightgbm: 'LightGBM',
    random_forest: 'Random Forest',
    logistic_regression: 'Logistic Regression',
    ensemble: 'Ensemble',
  };

  // Fold scores chart data
  const foldData = [];
  const maxFolds = Math.max(...modelEntries.map(([_, r]) => r.fold_scores?.length || 0), 0);
  for (let i = 0; i < maxFolds; i++) {
    const row = { fold: `Fold ${i + 1}` };
    modelEntries.forEach(([name, result]) => {
      if (result.fold_scores?.[i] !== undefined) {
        row[name] = result.fold_scores[i];
      }
    });
    foldData.push(row);
  }

  return (
    <section className="mb-12">
      <SectionHeader icon={Cpu} title="Model Training" subtitle="5-fold stratified cross-validation with multiple classifiers" />

      {/* Model cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {modelEntries.map(([name, result]) => {
          const isEnsemble = name === 'ensemble';
          const isBest = name === DATA.best_model;
          return (
            <div key={name} className={`border rounded-xl p-4 ${isBest ? 'border-brand-500 bg-brand-50 ring-2 ring-brand-200' : 'border-gray-200'}`}>
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-bold text-gray-800">{modelLabels[name] || name}</h4>
                {isBest && <Badge text="Best" variant="success" />}
              </div>
              <div className="text-3xl font-bold mb-1" style={{ color: modelColors[name] || '#666' }}>
                {(result.mean_balanced_accuracy || 0).toFixed(4)}
              </div>
              <div className="text-xs text-gray-500 mb-3">Balanced Accuracy (OOF)</div>
              {!isEnsemble && result.params && (
                <div className="text-xs text-gray-400 space-y-0.5">
                  {Object.entries(result.params).slice(0, 4).map(([k, v]) => (
                    <div key={k} className="truncate"><span className="font-mono">{k}</span>: {String(v)}</div>
                  ))}
                </div>
              )}
              {isEnsemble && result.params?.weights && (
                <div className="text-xs text-gray-400 space-y-0.5">
                  <div className="font-medium">Weights (BA-weighted):</div>
                  {Object.entries(result.params.weights).map(([k, v]) => (
                    <div key={k} className="truncate">{modelLabels[k] || k}: {v}</div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Fold scores chart */}
      {foldData.length > 0 && (
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Per-Fold Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={foldData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="fold" />
              <YAxis domain={['auto', 'auto']} />
              <Tooltip />
              <Legend />
              {modelEntries.filter(([n]) => n !== 'ensemble').map(([name]) => (
                <Line key={name} type="monotone" dataKey={name} name={modelLabels[name] || name}
                  stroke={modelColors[name]} strokeWidth={2} dot={{ r: 4 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

// ─── Section 4: Results & Validation ───────────────────────────────────

function ResultsSection() {
  const models = DATA.models || {};
  const review = DATA.review || {};
  const modelEntries = Object.entries(models);
  const [selectedModel, setSelectedModel] = useState(DATA.best_model || modelEntries[0]?.[0] || '');

  const modelLabels = {
    xgboost: 'XGBoost', lightgbm: 'LightGBM', random_forest: 'Random Forest',
    logistic_regression: 'Logistic Regression', ensemble: 'Ensemble',
  };

  const selected = models[selectedModel] || {};
  const cm = selected.metrics?.confusion_matrix || [];
  const report = selected.metrics?.classification_report || {};
  const classLabels = DATA.competition?.classes || ['Low', 'Medium', 'High'];

  // Comparison table data
  const comparisonData = modelEntries.map(([name, result]) => ({
    name: modelLabels[name] || name,
    balanced_accuracy: result.mean_balanced_accuracy || 0,
  })).sort((a, b) => b.balanced_accuracy - a.balanced_accuracy);

  // Per-class metrics for selected model
  const classMetrics = classLabels.map(label => {
    const r = report[label] || report[String(classLabels.indexOf(label))] || {};
    return {
      class: label,
      precision: r.precision || 0,
      recall: r.recall || 0,
      f1: r['f1-score'] || 0,
    };
  }).filter(m => m.precision > 0 || m.recall > 0);

  return (
    <section className="mb-12">
      <SectionHeader icon={ClipboardCheck} title="Results & Validation" subtitle="Model comparison, confusion matrices, and review" />

      {/* Model selector */}
      <div className="flex flex-wrap gap-2 mb-6">
        {modelEntries.map(([name]) => (
          <button key={name} onClick={() => setSelectedModel(name)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedModel === name
                ? 'bg-brand-600 text-white shadow-md'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}>
            {modelLabels[name] || name}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Comparison bar chart */}
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Model Comparison</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={comparisonData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 1]} />
              <YAxis type="category" dataKey="name" width={120} />
              <Tooltip formatter={(v) => v.toFixed(5)} />
              <Bar dataKey="balanced_accuracy" name="Balanced Accuracy" fill="#22c55e" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Confusion matrix */}
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">
            Confusion Matrix — {modelLabels[selectedModel] || selectedModel}
          </h3>
          {cm.length > 0 ? (
            <div className="inline-block">
              <div className="grid grid-cols-4 gap-0 text-sm">
                <div className="p-2 font-medium text-gray-500">Pred →</div>
                {classLabels.map(l => <div key={l} className="p-2 font-medium text-center text-gray-600">{l}</div>)}
                {cm.map((row, i) => (
                  <React.Fragment key={i}>
                    <div className="p-2 font-medium text-gray-600">{classLabels[i]}</div>
                    {row.map((val, j) => {
                      const isCorrect = i === j;
                      const maxVal = Math.max(...cm.flat());
                      const intensity = maxVal > 0 ? val / maxVal : 0;
                      return (
                        <div key={j} className={`p-2 text-center font-mono rounded ${
                          isCorrect ? 'font-bold' : ''
                        }`} style={{
                          backgroundColor: isCorrect
                            ? `rgba(34, 197, 94, ${0.1 + intensity * 0.4})`
                            : val > 0 ? `rgba(239, 68, 68, ${0.05 + intensity * 0.2})` : 'transparent'
                        }}>
                          {val}
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-gray-400 text-sm">Run pipeline to see confusion matrix</div>
          )}
        </div>
      </div>

      {/* Per-class metrics */}
      {classMetrics.length > 0 && (
        <div className="mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">Per-Class Metrics — {modelLabels[selectedModel]}</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={classMetrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="class" />
              <YAxis domain={[0, 1]} />
              <Tooltip formatter={(v) => v.toFixed(4)} />
              <Legend />
              <Bar dataKey="precision" fill="#3b82f6" />
              <Bar dataKey="recall" fill="#22c55e" />
              <Bar dataKey="f1" name="F1" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Review verdict */}
      <div className="border rounded-xl p-5">
        <h3 className="font-semibold text-gray-800 mb-3">Reviewer Verdict</h3>
        <div className="flex items-center gap-2 mb-4">
          {review.verdict === 'APPROVE' && <><CheckCircle2 className="w-6 h-6 text-green-500" /><span className="font-bold text-green-600">APPROVED</span></>}
          {review.verdict === 'CONDITIONAL_APPROVE' && <><AlertTriangle className="w-6 h-6 text-amber-500" /><span className="font-bold text-amber-600">CONDITIONALLY APPROVED</span></>}
          {review.verdict === 'REJECT' && <><XCircle className="w-6 h-6 text-red-500" /><span className="font-bold text-red-600">REJECTED</span></>}
          {!review.verdict || review.verdict === 'PENDING' && <span className="text-gray-400">Pending — run pipeline first</span>}
        </div>
        {review.approvals?.length > 0 && (
          <div className="space-y-1 mb-3">
            {review.approvals.map((a, i) => (
              <div key={i} className="flex items-start gap-2 text-sm text-green-600">
                <CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />{a}
              </div>
            ))}
          </div>
        )}
        {review.warnings?.length > 0 && (
          <div className="space-y-1 mb-3">
            {review.warnings.map((w, i) => (
              <div key={i} className="flex items-start gap-2 text-sm text-amber-600">
                <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />{w}
              </div>
            ))}
          </div>
        )}
        {review.issues?.length > 0 && (
          <div className="space-y-1">
            {review.issues.map((issue, i) => (
              <div key={i} className="flex items-start gap-2 text-sm text-red-600">
                <XCircle className="w-4 h-4 mt-0.5 shrink-0" />{issue}
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

// ─── Section 5: Summary & Interpretation ───────────────────────────────

function SummarySection() {
  const bestModel = DATA.best_model;
  const models = DATA.models || {};
  const bestResult = models[bestModel] || {};
  const importance = bestResult.feature_importance || {};
  const importanceData = Object.entries(importance).slice(0, 15).map(([name, value]) => ({
    name, value: Number(value.toFixed(4))
  })).reverse();

  const modelLabels = {
    xgboost: 'XGBoost', lightgbm: 'LightGBM', random_forest: 'Random Forest',
    logistic_regression: 'Logistic Regression', ensemble: 'Ensemble',
  };

  const pipelineSteps = DATA.pipeline?.steps || [];

  return (
    <section className="mb-12">
      <SectionHeader icon={Sun} title="Summary & Interpretation" subtitle="Key findings and feature importance" />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <StatCard label="Best Model" value={modelLabels[bestModel] || bestModel || '—'} color="brand" />
        <StatCard label="Balanced Accuracy" value={(DATA.best_score || 0).toFixed(4)} color="blue" />
        <StatCard label="Pipeline Time" value={`${DATA.elapsed_seconds || 0}s`} color="amber" />
      </div>

      {/* Feature importance */}
      {importanceData.length > 0 && (
        <div className="mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">Top Feature Importance ({modelLabels[bestModel] || bestModel})</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={importanceData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={180} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="value" name="Importance" fill="#22c55e" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Pipeline overview */}
      {pipelineSteps.length > 0 && (
        <div className="mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">Pipeline Overview</h3>
          <div className="flex flex-wrap gap-2">
            {pipelineSteps.map((step) => (
              <div key={step.step} className="flex items-center gap-2 bg-gray-100 rounded-full px-4 py-2 text-sm">
                <span className="bg-brand-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">{step.step}</span>
                <span className="text-gray-700">{step.name}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Prediction distribution comparison */}
      {DATA.review?.prediction_distribution && Object.keys(DATA.review.prediction_distribution).length > 0 && (
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Prediction vs Training Distribution</h3>
          <div className="grid grid-cols-3 gap-4">
            {(DATA.competition?.classes || []).map(label => {
              const trainPct = (DATA.review.training_distribution?.[label] || 0) * 100;
              const predPct = (DATA.review.prediction_distribution?.[label] || 0) * 100;
              return (
                <div key={label} className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="font-medium text-gray-800 mb-2">{label}</div>
                  <div className="text-sm">
                    <span className="text-blue-600">Train: {trainPct.toFixed(1)}%</span>
                    {' / '}
                    <span className="text-green-600">Pred: {predPct.toFixed(1)}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────

export default function App() {
  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-gradient-to-r from-brand-700 to-brand-900 text-white py-8 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center gap-3 mb-2">
            <Droplets className="w-8 h-8" />
            <h1 className="text-3xl font-bold" style={{ fontFamily: 'Georgia, serif' }}>Irrigation Need Prediction</h1>
          </div>
          <p className="text-brand-100 text-lg">Kaggle Playground Series S6E4 — Multi-class Classification Dashboard</p>
          <div className="flex gap-4 mt-4 text-sm text-brand-200">
            <span>Target: Irrigation_Need</span>
            <span>|</span>
            <span>Classes: Low, Medium, High</span>
            <span>|</span>
            <span>Metric: Balanced Accuracy</span>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-6xl mx-auto px-6 py-10">
        <IntroductionSection />
        <DataCleaningSection />
        <ModelTrainingSection />
        <ResultsSection />
        <SummarySection />
      </main>

      {/* Footer */}
      <footer className="bg-gray-50 border-t py-6 px-6">
        <div className="max-w-6xl mx-auto text-center text-sm text-gray-400">
          Built with AutoKaggle 4-Agent Pipeline — React + Vite + Tailwind CSS + Recharts
        </div>
      </footer>
    </div>
  );
}
