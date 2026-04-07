import React, { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, LineChart, Line, Cell
} from 'recharts';
import {
  Droplets, Sun, Cloud, Leaf, BarChart3, CheckCircle2,
  AlertTriangle, XCircle, ChevronDown, ChevronUp, Database,
  Cpu, FlaskConical, ClipboardCheck, TrendingUp, ArrowUpRight, ArrowDownRight
} from 'lucide-react';

const CLASS_COLORS = { Low: '#22c55e', Medium: '#f59e0b', High: '#ef4444' };

const MODEL_LABELS = {
  xgboost: 'XGBoost', lightgbm: 'LightGBM', random_forest: 'Random Forest',
  logistic_regression: 'Logistic Regression', ensemble: 'Ensemble',
  xgboost_tuned: 'XGBoost (Tuned)', lightgbm_tuned: 'LightGBM (Tuned)',
  random_forest_tuned: 'Random Forest (Tuned)', extra_trees: 'Extra Trees',
  gradient_boosting: 'Gradient Boosting', stacking_ensemble: 'Stacking Ensemble',
  weighted_ensemble: 'Weighted Ensemble',
};

const MODEL_COLORS = {
  xgboost: '#ef4444', lightgbm: '#22c55e', random_forest: '#3b82f6',
  logistic_regression: '#f59e0b', ensemble: '#8b5cf6',
  xgboost_tuned: '#dc2626', lightgbm_tuned: '#16a34a',
  random_forest_tuned: '#2563eb', extra_trees: '#0891b2',
  gradient_boosting: '#d97706', stacking_ensemble: '#7c3aed',
  weighted_ensemble: '#9333ea',
};

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
    purple: 'bg-purple-50 border-purple-200 text-purple-700',
  };
  return (
    <div className={`rounded-xl border p-4 ${colorMap[color]}`}>
      <div className="text-sm font-medium opacity-75">{label}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
      {sub && <div className="text-xs mt-1 opacity-60">{sub}</div>}
    </div>
  );
}

function CodeBlock({ code }) {
  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden">
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
    purple: 'bg-purple-100 text-purple-700',
  };
  return <span className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${styles[variant]}`}>{text}</span>;
}

// ─── Section 1: Introduction ───────────────────────────────────────────

function IntroductionSection({ data }) {
  const cats = data.eda?.feature_categories || {};
  const catIcons = { soil: Droplets, weather: Cloud, crop: Leaf, field: BarChart3 };
  const catLabels = { soil: 'Soil Properties', weather: 'Weather Data', crop: 'Crop Info', field: 'Field Characteristics' };
  const classDist = data.eda?.class_distribution?.counts || {};
  const classData = Object.entries(classDist).map(([name, value]) => ({ name, value, fill: CLASS_COLORS[name] || '#888' }));

  return (
    <section className="mb-12">
      <SectionHeader icon={Database} title="Introduction" subtitle="Dataset overview and competition context" />
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <StatCard label="Competition" value={data.competition?.name || 'S6E4'} sub="Playground Series" />
        <StatCard label="Train Rows" value={(data.dataset?.train_shape?.[0] || 0).toLocaleString()} sub={`${data.dataset?.train_shape?.[1] || 0} columns`} color="blue" />
        <StatCard label="Test Rows" value={(data.dataset?.test_shape?.[0] || 0).toLocaleString()} color="blue" />
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
                  {classData.map((entry, i) => <Cell key={i} fill={entry.fill} />)}
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

function DataCleaningSection({ data }) {
  const steps = data.pipeline?.steps || [];
  const sample = data.eda?.sample_data || { columns: [], rows: [] };
  const missing = data.eda?.missing_values || {};
  const numStats = data.eda?.numeric_stats || {};
  const [showAllStats, setShowAllStats] = useState(false);
  const statEntries = Object.entries(numStats);
  const visibleStats = showAllStats ? statEntries : statEntries.slice(0, 6);

  return (
    <section className="mb-12">
      <SectionHeader icon={FlaskConical} title="Data Cleaning & Feature Engineering" subtitle="Preprocessing pipeline and feature analysis" />
      {sample.columns.length > 0 && (
        <div className="mb-6">
          <h3 className="font-semibold text-gray-800 mb-2">Raw Data Sample</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm border border-gray-200 rounded-lg overflow-hidden">
              <thead className="bg-gray-100">
                <tr>{sample.columns.map(col => <th key={col} className="px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap">{col}</th>)}</tr>
              </thead>
              <tbody>
                {sample.rows.map((row, i) => (
                  <tr key={i} className={i % 2 ? 'bg-gray-50' : ''}>
                    {row.map((cell, j) => <td key={j} className="px-3 py-1.5 whitespace-nowrap text-gray-700">{cell != null ? String(cell) : '\u2014'}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-800 mb-2">Missing Values</h3>
          {missing.train_total === 0 && missing.test_total === 0 ? (
            <div className="flex items-center gap-2 text-green-600"><CheckCircle2 className="w-5 h-5" /><span>No missing values in train or test</span></div>
          ) : (
            <div className="space-y-1 text-sm">
              <div>Train: {missing.train_total} missing</div>
              <div>Test: {missing.test_total} missing</div>
            </div>
          )}
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold text-gray-800 mb-2">Numeric Feature Stats</h3>
          {visibleStats.length > 0 ? (
            <div className="space-y-1 text-sm">
              {visibleStats.map(([feat, stats]) => (
                <div key={feat} className="flex justify-between">
                  <span className="text-gray-600">{feat}</span>
                  <span className="font-mono text-gray-800">mean={Number(stats.mean).toFixed(2)} std={Number(stats.std).toFixed(2)}</span>
                </div>
              ))}
              {statEntries.length > 6 && (
                <button onClick={() => setShowAllStats(!showAllStats)} className="text-brand-600 text-xs flex items-center gap-1 mt-2">
                  {showAllStats ? <><ChevronUp className="w-3 h-3" /> Show less</> : <><ChevronDown className="w-3 h-3" /> Show all ({statEntries.length})</>}
                </button>
              )}
            </div>
          ) : <div className="text-gray-400 text-sm">Run pipeline to see stats</div>}
        </div>
      </div>
      {steps.length > 0 && (
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Preprocessing Pipeline ({steps.length} steps)</h3>
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

function ModelTrainingSection({ data }) {
  const models = data.models || {};
  const modelEntries = Object.entries(models);

  const foldData = [];
  const cvModels = modelEntries.filter(([n, r]) => r.fold_scores?.length === 5);
  const maxFolds = 5;
  for (let i = 0; i < maxFolds; i++) {
    const row = { fold: `Fold ${i + 1}` };
    cvModels.forEach(([name, result]) => { row[name] = result.fold_scores?.[i]; });
    foldData.push(row);
  }

  return (
    <section className="mb-12">
      <SectionHeader icon={Cpu} title="Model Training" subtitle="5-fold stratified cross-validation with multiple classifiers" />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {modelEntries.map(([name, result]) => {
          const isBest = name === data.best_model;
          const isEnsemble = name.includes('ensemble');
          return (
            <div key={name} className={`border rounded-xl p-4 ${isBest ? 'border-brand-500 bg-brand-50 ring-2 ring-brand-200' : 'border-gray-200'}`}>
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-bold text-gray-800 text-sm">{MODEL_LABELS[name] || name}</h4>
                {isBest && <Badge text="Best" variant="success" />}
              </div>
              <div className="text-3xl font-bold mb-1" style={{ color: MODEL_COLORS[name] || '#666' }}>
                {(result.mean_balanced_accuracy || 0).toFixed(4)}
              </div>
              <div className="text-xs text-gray-500 mb-3">Balanced Accuracy (OOF)</div>
              {isEnsemble && result.params?.weights && (
                <div className="text-xs text-gray-400 space-y-0.5">
                  {Object.entries(result.params.weights).map(([k, v]) => (
                    <div key={k} className="truncate">{MODEL_LABELS[k] || k}: {v}</div>
                  ))}
                </div>
              )}
              {isEnsemble && result.params?.method && !result.params?.weights && (
                <div className="text-xs text-gray-400">Method: {result.params.method}</div>
              )}
              {!isEnsemble && result.fold_scores?.length === 5 && (
                <div className="text-xs text-gray-400">Folds: {result.fold_scores.map(s => s.toFixed(4)).join(', ')}</div>
              )}
            </div>
          );
        })}
      </div>
      {foldData.length > 0 && cvModels.length > 0 && (
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Per-Fold Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={foldData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="fold" />
              <YAxis domain={['auto', 'auto']} />
              <Tooltip formatter={(v) => typeof v === 'number' ? v.toFixed(5) : v} />
              <Legend />
              {cvModels.map(([name]) => (
                <Line key={name} type="monotone" dataKey={name} name={MODEL_LABELS[name] || name}
                  stroke={MODEL_COLORS[name] || '#888'} strokeWidth={2} dot={{ r: 3 }} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

// ─── Section 4: Results & Validation ───────────────────────────────────

function ResultsSection({ data }) {
  const models = data.models || {};
  const review = data.review || {};
  const modelEntries = Object.entries(models);
  const [selectedModel, setSelectedModel] = useState(data.best_model || '');
  const selected = models[selectedModel] || {};
  const cm = selected.metrics?.confusion_matrix || [];
  const report = selected.metrics?.classification_report || {};
  const classLabels = data.competition?.classes || ['Low', 'Medium', 'High'];

  const comparisonData = modelEntries.map(([name, result]) => ({
    name: MODEL_LABELS[name] || name,
    balanced_accuracy: result.mean_balanced_accuracy || 0,
  })).sort((a, b) => b.balanced_accuracy - a.balanced_accuracy);

  const classMetrics = classLabels.map(label => {
    const r = report[label] || report[String(classLabels.indexOf(label))] || {};
    return { class: label, precision: r.precision || 0, recall: r.recall || 0, f1: r['f1-score'] || 0 };
  }).filter(m => m.precision > 0 || m.recall > 0);

  return (
    <section className="mb-12">
      <SectionHeader icon={ClipboardCheck} title="Results & Validation" subtitle="Model comparison, confusion matrices, and review" />
      <div className="flex flex-wrap gap-2 mb-6">
        {modelEntries.map(([name]) => (
          <button key={name} onClick={() => setSelectedModel(name)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${selectedModel === name ? 'bg-brand-600 text-white shadow-md' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}>
            {MODEL_LABELS[name] || name}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Model Comparison</h3>
          <ResponsiveContainer width="100%" height={Math.max(250, comparisonData.length * 35)}>
            <BarChart data={comparisonData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 1]} />
              <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => v.toFixed(5)} />
              <Bar dataKey="balanced_accuracy" name="Balanced Accuracy" fill="#22c55e" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Confusion Matrix — {MODEL_LABELS[selectedModel] || selectedModel}</h3>
          {cm.length > 0 ? (
            <div className="inline-block">
              <div className="grid grid-cols-4 gap-0 text-sm">
                <div className="p-2 font-medium text-gray-500">Pred &rarr;</div>
                {classLabels.map(l => <div key={l} className="p-2 font-medium text-center text-gray-600">{l}</div>)}
                {cm.map((row, i) => (
                  <React.Fragment key={i}>
                    <div className="p-2 font-medium text-gray-600">{classLabels[i]}</div>
                    {row.map((val, j) => {
                      const isCorrect = i === j;
                      const maxVal = Math.max(...cm.flat());
                      const intensity = maxVal > 0 ? val / maxVal : 0;
                      return (
                        <div key={j} className={`p-2 text-center font-mono rounded ${isCorrect ? 'font-bold' : ''}`} style={{
                          backgroundColor: isCorrect ? `rgba(34, 197, 94, ${0.1 + intensity * 0.4})` : val > 0 ? `rgba(239, 68, 68, ${0.05 + intensity * 0.2})` : 'transparent'
                        }}>{val.toLocaleString()}</div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
            </div>
          ) : <div className="text-gray-400 text-sm">Run pipeline to see confusion matrix</div>}
        </div>
      </div>
      {classMetrics.length > 0 && (
        <div className="mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">Per-Class Metrics — {MODEL_LABELS[selectedModel] || selectedModel}</h3>
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
      <div className="border rounded-xl p-5">
        <h3 className="font-semibold text-gray-800 mb-3">Reviewer Verdict</h3>
        <div className="flex items-center gap-2 mb-4">
          {review.verdict === 'APPROVE' && <><CheckCircle2 className="w-6 h-6 text-green-500" /><span className="font-bold text-green-600">APPROVED</span></>}
          {review.verdict === 'CONDITIONAL_APPROVE' && <><AlertTriangle className="w-6 h-6 text-amber-500" /><span className="font-bold text-amber-600">CONDITIONALLY APPROVED</span></>}
          {review.verdict === 'REJECT' && <><XCircle className="w-6 h-6 text-red-500" /><span className="font-bold text-red-600">REJECTED</span></>}
          {(!review.verdict || review.verdict === 'PENDING') && <span className="text-gray-400">Pending</span>}
        </div>
        {review.approvals?.length > 0 && <div className="space-y-1 mb-3">{review.approvals.map((a, i) => <div key={i} className="flex items-start gap-2 text-sm text-green-600"><CheckCircle2 className="w-4 h-4 mt-0.5 shrink-0" />{a}</div>)}</div>}
        {review.warnings?.length > 0 && <div className="space-y-1 mb-3">{review.warnings.map((w, i) => <div key={i} className="flex items-start gap-2 text-sm text-amber-600"><AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />{w}</div>)}</div>}
        {review.issues?.length > 0 && <div className="space-y-1">{review.issues.map((issue, i) => <div key={i} className="flex items-start gap-2 text-sm text-red-600"><XCircle className="w-4 h-4 mt-0.5 shrink-0" />{issue}</div>)}</div>}
      </div>
    </section>
  );
}

// ─── Section 5: Iterative Improvements ─────────────────────────────────

function ImprovementsSection({ data }) {
  const imp = data.improvement;
  if (!imp) return null;

  const isImproved = imp.score_change > 0;
  const recommendations = imp.recommendations_applied || [];
  const analysis = imp.research_analysis || {};

  const versionData = [
    { version: 'V1 (Baseline)', score: imp.previous_best_score, model: imp.previous_best_model },
    { version: 'V2 (Improved)', score: imp.improved_best_score, model: imp.improved_best_model },
  ];

  return (
    <section className="mb-12">
      <SectionHeader icon={TrendingUp} title="Iterative Improvements" subtitle="Version comparison and changes applied" />

      {/* Score comparison */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <StatCard label="V1 Baseline" value={(imp.previous_best_score || 0).toFixed(5)} sub={MODEL_LABELS[imp.previous_best_model] || imp.previous_best_model} color="blue" />
        <StatCard label="V2 Improved" value={(imp.improved_best_score || 0).toFixed(5)} sub={MODEL_LABELS[imp.improved_best_model] || imp.improved_best_model} color="brand" />
        <div className={`rounded-xl border p-4 ${isImproved ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <div className="text-sm font-medium opacity-75">Score Change</div>
          <div className={`text-2xl font-bold mt-1 flex items-center gap-2 ${isImproved ? 'text-green-700' : 'text-red-700'}`}>
            {isImproved ? <ArrowUpRight className="w-6 h-6" /> : <ArrowDownRight className="w-6 h-6" />}
            {isImproved ? '+' : ''}{(imp.score_change || 0).toFixed(5)}
          </div>
          <div className="text-xs mt-1 opacity-60">{isImproved ? 'Improvement' : 'Regression'}</div>
        </div>
      </div>

      {/* Version chart */}
      <div className="mb-6">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={versionData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="version" />
            <YAxis domain={[0.9, 1]} />
            <Tooltip formatter={(v) => v.toFixed(5)} />
            <Bar dataKey="score" name="Balanced Accuracy">
              {versionData.map((_, i) => <Cell key={i} fill={i === 0 ? '#3b82f6' : '#22c55e'} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Changes applied */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Changes Applied (V1 → V2)</h3>
          <div className="space-y-2">
            {recommendations.map((rec, i) => (
              <div key={i} className="flex items-start gap-2 p-3 bg-gray-50 rounded-lg">
                <CheckCircle2 className="w-4 h-4 text-brand-600 mt-0.5 shrink-0" />
                <span className="text-sm text-gray-700">{rec}</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Research Analysis</h3>
          {analysis.class_imbalance && (
            <div className="p-3 bg-gray-50 rounded-lg mb-3">
              <div className="text-sm font-medium text-gray-700 mb-1">Class Imbalance</div>
              <div className="space-y-1">
                {Object.entries(analysis.class_imbalance).map(([cls, pct]) => (
                  <div key={cls} className="flex justify-between text-sm">
                    <span className="text-gray-600">{cls}</span>
                    <span className="font-mono">{(pct * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {analysis.per_class_accuracy && (
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="text-sm font-medium text-gray-700 mb-1">Per-Class Accuracy (V1)</div>
              <div className="space-y-1">
                {Object.entries(analysis.per_class_accuracy).map(([cls, acc]) => (
                  <div key={cls} className="flex justify-between text-sm">
                    <span className="text-gray-600">{cls}</span>
                    <span className={`font-mono ${acc < 0.9 ? 'text-red-600 font-bold' : 'text-green-600'}`}>{(acc * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

// ─── Section 6: Summary & Interpretation ───────────────────────────────

function SummarySection({ data }) {
  const bestModel = data.best_model;
  const models = data.models || {};
  const bestResult = models[bestModel] || {};
  const importance = bestResult.feature_importance || {};
  const importanceData = Object.entries(importance).slice(0, 15).map(([name, value]) => ({
    name, value: Number(Number(value).toFixed(4))
  })).reverse();
  const pipelineSteps = data.pipeline?.steps || [];

  return (
    <section className="mb-12">
      <SectionHeader icon={Sun} title="Summary & Interpretation" subtitle="Key findings and feature importance" />
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <StatCard label="Best Model" value={MODEL_LABELS[bestModel] || bestModel || '\u2014'} color="brand" />
        <StatCard label="Balanced Accuracy" value={(data.best_score || 0).toFixed(5)} color="blue" />
        <StatCard label="Pipeline Time" value={`${data.elapsed_seconds || 0}s`} color="amber" />
      </div>
      {importanceData.length > 0 && (
        <div className="mb-6">
          <h3 className="font-semibold text-gray-800 mb-3">Top Feature Importance ({MODEL_LABELS[bestModel] || bestModel})</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={importanceData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis type="category" dataKey="name" width={200} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="value" name="Importance" fill="#22c55e" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
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
      {data.review?.prediction_distribution && Object.keys(data.review.prediction_distribution).length > 0 && (
        <div>
          <h3 className="font-semibold text-gray-800 mb-3">Prediction vs Training Distribution</h3>
          <div className="grid grid-cols-3 gap-4">
            {(data.competition?.classes || []).map(label => {
              const trainPct = (data.review.training_distribution?.[label] || 0) * 100;
              const predPct = (data.review.prediction_distribution?.[label] || 0) * 100;
              return (
                <div key={label} className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="font-medium text-gray-800 mb-2">{label}</div>
                  <div className="text-sm"><span className="text-blue-600">Train: {trainPct.toFixed(1)}%</span>{' / '}<span className="text-green-600">Pred: {predPct.toFixed(1)}%</span></div>
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
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/results.json')
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const ct = r.headers.get('content-type') || '';
        if (!ct.includes('json')) {
          return r.text().then(t => { throw new Error(`Expected JSON but got ${ct}: ${t.slice(0, 100)}`); });
        }
        return r.json();
      })
      .then(d => { setData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return (
    <div className="min-h-screen bg-white flex items-center justify-center">
      <div className="text-center">
        <Droplets className="w-12 h-12 text-brand-600 mx-auto mb-4 animate-pulse" />
        <p className="text-gray-500">Loading pipeline results...</p>
      </div>
    </div>
  );

  if (error || !data || !data.competition) return (
    <div className="min-h-screen bg-white flex items-center justify-center">
      <div className="text-center max-w-md">
        <AlertTriangle className="w-12 h-12 text-amber-500 mx-auto mb-4" />
        <h2 className="text-xl font-bold text-gray-800 mb-2">No Pipeline Results Yet</h2>
        <p className="text-gray-500 mb-4">Run the pipeline first to generate results:</p>
        <CodeBlock code="python3 model_pipeline.py" />
        {error && <p className="text-red-400 text-sm mt-4">Error: {error}</p>}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-white">
      <header className="bg-gradient-to-r from-brand-700 to-brand-900 text-white py-8 px-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center gap-3 mb-2">
            <Droplets className="w-8 h-8" />
            <h1 className="text-3xl font-bold" style={{ fontFamily: 'Georgia, serif' }}>Irrigation Need Prediction</h1>
          </div>
          <p className="text-brand-100 text-lg">Kaggle Playground Series S6E4 — Multi-class Classification Dashboard</p>
          <div className="flex flex-wrap gap-4 mt-4 text-sm text-brand-200">
            <span>Target: Irrigation_Need</span>
            <span>|</span>
            <span>Classes: Low, Medium, High</span>
            <span>|</span>
            <span>Metric: Balanced Accuracy</span>
            <span>|</span>
            <span className="font-bold text-white">Best Score: {(data.best_score || 0).toFixed(5)}</span>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10">
        <IntroductionSection data={data} />
        <DataCleaningSection data={data} />
        <ModelTrainingSection data={data} />
        <ResultsSection data={data} />
        {data.improvement && <ImprovementsSection data={data} />}
        <SummarySection data={data} />
      </main>

      <footer className="bg-gray-50 border-t py-6 px-6">
        <div className="max-w-6xl mx-auto text-center text-sm text-gray-400">
          Built with AutoKaggle 4-Agent Pipeline — React + Vite + Tailwind CSS + Recharts
        </div>
      </footer>
    </div>
  );
}
