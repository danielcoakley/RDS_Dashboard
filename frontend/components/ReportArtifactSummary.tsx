"use client";

import ReactECharts from "echarts-for-react";

type ReportArtifactSummaryProps = {
  content: Record<string, unknown> | null;
};

type NumericMetric = {
  key: string;
  label: string;
  value: number;
};

type SourcePackageSummary = {
  uploadCount: number | null;
  categories: string[];
  executionInput: string;
};

type LegacyBaselineSummary = {
  energyRecords: number | null;
  meterCount: number | null;
  mappedMeterCount: number | null;
  hddRecords: number | null;
  cddRecords: number | null;
  weatherModelCount: number | null;
  estimatedSavings: number | null;
  seuCategories: string[];
};

type WeatherModelSummary = {
  meter: string;
  utility: string;
  seuCategory: string;
  normalization: string;
  baselineRecords: number | null;
  comparisonRecords: number | null;
  actualConsumption: number | null;
  predictedConsumption: number | null;
  estimatedSavings: number | null;
  r2: number | null;
};

type SkippedWeatherModel = {
  meter: string;
  utility: string;
  seuCategory: string;
  reason: string;
};

type WeatherNormalizationSummary = {
  status: string;
  baselineYear: number | null;
  comparisonYear: number | null;
  modelCount: number | null;
  skippedMeterCount: number | null;
  estimatedSavings: number | null;
  models: WeatherModelSummary[];
  skippedMeters: SkippedWeatherModel[];
};

function textValue(content: Record<string, unknown> | null, key: string): string {
  const value = content?.[key];
  return typeof value === "string" && value.trim() ? value : "Not recorded";
}

function numericValue(content: Record<string, unknown> | null, key: string): number | null {
  const value = content?.[key];
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function formatNumber(value: number | null): string {
  if (value === null) {
    return "Not recorded";
  }
  return new Intl.NumberFormat("en-GB", {
    maximumFractionDigits: value < 10 ? 2 : 0
  }).format(value);
}

function labelFromKey(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function numberFromRecord(record: Record<string, unknown>, key: string): number | null {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringFromRecord(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  return typeof value === "string" && value.trim() ? value : "Not recorded";
}

function formatSignedNumber(value: number | null): string {
  if (value === null) {
    return "Not recorded";
  }
  const formatted = formatNumber(Math.abs(value));
  if (value === 0) {
    return formatted;
  }
  return `${value > 0 ? "+" : "-"}${formatted}`;
}

function sourcePackageSummary(content: Record<string, unknown> | null): SourcePackageSummary {
  const sourcePackage = content?.source_package;
  if (!sourcePackage || typeof sourcePackage !== "object" || Array.isArray(sourcePackage)) {
    return { uploadCount: null, categories: [], executionInput: "Not recorded" };
  }
  const packageRecord = sourcePackage as Record<string, unknown>;
  const analysisContext = packageRecord.analysis_context;
  const analysisRecord =
    analysisContext && typeof analysisContext === "object" && !Array.isArray(analysisContext)
      ? (analysisContext as Record<string, unknown>)
      : {};
  const uploadCount =
    typeof packageRecord.upload_count === "number" && Number.isFinite(packageRecord.upload_count)
      ? packageRecord.upload_count
      : null;
  const categories = Array.isArray(packageRecord.categories)
    ? packageRecord.categories.filter((category): category is string => typeof category === "string")
    : [];
  const executionInput =
    typeof analysisRecord.execution_input === "string" && analysisRecord.execution_input.trim()
      ? labelFromKey(analysisRecord.execution_input)
      : "Not recorded";
  return { uploadCount, categories, executionInput };
}

function legacyBaselineSummary(content: Record<string, unknown> | null): LegacyBaselineSummary | null {
  const legacyBaseline = content?.legacy_baseline;
  if (!legacyBaseline || typeof legacyBaseline !== "object" || Array.isArray(legacyBaseline)) {
    return null;
  }
  const legacyRecord = legacyBaseline as Record<string, unknown>;
  const weatherCoverage = legacyRecord.weather_coverage;
  const weatherRecord =
    weatherCoverage && typeof weatherCoverage === "object" && !Array.isArray(weatherCoverage)
      ? (weatherCoverage as Record<string, unknown>)
      : {};
  const hddCoverage = weatherRecord.hdd;
  const cddCoverage = weatherRecord.cdd;
  const hddRecord =
    hddCoverage && typeof hddCoverage === "object" && !Array.isArray(hddCoverage)
      ? (hddCoverage as Record<string, unknown>)
      : {};
  const cddRecord =
    cddCoverage && typeof cddCoverage === "object" && !Array.isArray(cddCoverage)
      ? (cddCoverage as Record<string, unknown>)
      : {};
  const weatherNormalization = legacyRecord.weather_normalization;
  const weatherNormalizationRecord =
    weatherNormalization && typeof weatherNormalization === "object" && !Array.isArray(weatherNormalization)
      ? (weatherNormalization as Record<string, unknown>)
      : {};
  const seuCategories = Array.isArray(legacyRecord.seu_categories)
    ? legacyRecord.seu_categories.filter((category): category is string => typeof category === "string")
    : [];

  return {
    energyRecords: numberFromRecord(legacyRecord, "energy_record_count"),
    meterCount: numberFromRecord(legacyRecord, "meter_count"),
    mappedMeterCount: numberFromRecord(legacyRecord, "mapped_meter_count"),
    hddRecords: numberFromRecord(hddRecord, "record_count"),
    cddRecords: numberFromRecord(cddRecord, "record_count"),
    weatherModelCount: numberFromRecord(weatherNormalizationRecord, "model_count"),
    estimatedSavings: numberFromRecord(weatherNormalizationRecord, "estimated_savings"),
    seuCategories
  };
}

function weatherNormalizationSummary(
  content: Record<string, unknown> | null
): WeatherNormalizationSummary | null {
  const legacyBaseline = content?.legacy_baseline;
  if (!legacyBaseline || typeof legacyBaseline !== "object" || Array.isArray(legacyBaseline)) {
    return null;
  }
  const legacyRecord = legacyBaseline as Record<string, unknown>;
  const weatherNormalization = legacyRecord.weather_normalization;
  if (
    !weatherNormalization ||
    typeof weatherNormalization !== "object" ||
    Array.isArray(weatherNormalization)
  ) {
    return null;
  }
  const weatherRecord = weatherNormalization as Record<string, unknown>;
  const models = Array.isArray(weatherRecord.models)
    ? weatherRecord.models.flatMap((model): WeatherModelSummary[] => {
        if (!model || typeof model !== "object" || Array.isArray(model)) {
          return [];
        }
        const modelRecord = model as Record<string, unknown>;
        return [
          {
            meter: stringFromRecord(modelRecord, "meter"),
            utility: stringFromRecord(modelRecord, "utility"),
            seuCategory: stringFromRecord(modelRecord, "seu_category"),
            normalization: stringFromRecord(modelRecord, "normalization"),
            baselineRecords: numberFromRecord(modelRecord, "baseline_record_count"),
            comparisonRecords: numberFromRecord(modelRecord, "comparison_record_count"),
            actualConsumption: numberFromRecord(modelRecord, "actual_consumption"),
            predictedConsumption: numberFromRecord(modelRecord, "predicted_consumption"),
            estimatedSavings: numberFromRecord(modelRecord, "estimated_savings"),
            r2: numberFromRecord(modelRecord, "r2")
          }
        ];
      })
    : [];
  const skippedMeters = Array.isArray(weatherRecord.skipped_meters)
    ? weatherRecord.skipped_meters.flatMap((skippedMeter): SkippedWeatherModel[] => {
        if (!skippedMeter || typeof skippedMeter !== "object" || Array.isArray(skippedMeter)) {
          return [];
        }
        const skippedRecord = skippedMeter as Record<string, unknown>;
        return [
          {
            meter: stringFromRecord(skippedRecord, "meter"),
            utility: stringFromRecord(skippedRecord, "utility"),
            seuCategory: stringFromRecord(skippedRecord, "seu_category"),
            reason: stringFromRecord(skippedRecord, "reason")
          }
        ];
      })
    : [];

  return {
    status: stringFromRecord(weatherRecord, "status"),
    baselineYear: numberFromRecord(weatherRecord, "baseline_year"),
    comparisonYear: numberFromRecord(weatherRecord, "comparison_year"),
    modelCount: numberFromRecord(weatherRecord, "model_count"),
    skippedMeterCount: numberFromRecord(weatherRecord, "skipped_meter_count"),
    estimatedSavings: numberFromRecord(weatherRecord, "estimated_savings"),
    models,
    skippedMeters
  };
}

function numericMetrics(content: Record<string, unknown> | null): NumericMetric[] {
  if (!content) {
    return [];
  }
  return Object.entries(content)
    .flatMap(([key, value]) => {
      const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
      if (!Number.isFinite(parsed)) {
        return [];
      }
      return [{ key, label: labelFromKey(key), value: parsed }];
    })
    .slice(0, 8);
}

export function ReportArtifactSummary({ content }: ReportArtifactSummaryProps) {
  const client = textValue(content, "client");
  const site = textValue(content, "site");
  const totalRecords = numericValue(content, "total_records");
  const seuConsumption = numericValue(content, "seu_total_consumption");
  const intensity =
    totalRecords !== null && totalRecords > 0 && seuConsumption !== null
      ? seuConsumption / totalRecords
      : null;
  const sourcePackage = sourcePackageSummary(content);
  const legacyBaseline = legacyBaselineSummary(content);
  const weatherNormalization = weatherNormalizationSummary(content);
  const weatherModels = weatherNormalization?.models ?? [];
  const metrics = numericMetrics(content);
  const chartOption = {
    color: ["#0f9cae"],
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      valueFormatter: (value: number | string) =>
        typeof value === "number" ? formatNumber(value) : String(value)
    },
    grid: {
      top: 28,
      right: 18,
      bottom: 54,
      left: 64
    },
    xAxis: {
      type: "category",
      data: metrics.map((metric) => metric.label),
      axisLabel: {
        interval: 0,
        rotate: metrics.length > 3 ? 24 : 0,
        width: 96,
        overflow: "break"
      }
    },
    yAxis: {
      type: "value"
    },
    series: [
      {
        name: "Value",
        type: "bar",
        data: metrics.map((metric) => metric.value),
        barMaxWidth: 42,
        label: {
          show: true,
          position: "top",
          formatter: ({ value }: { value?: number | string }) =>
            typeof value === "number" ? formatNumber(value) : ""
        }
      }
    ]
  };
  const weatherModelChartOption = {
    color: ["#0f9cae", "#f59e0b", "#2f855a"],
    legend: {
      top: 0
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      valueFormatter: (value: number | string) =>
        typeof value === "number" ? formatNumber(value) : String(value)
    },
    grid: {
      top: 48,
      right: 18,
      bottom: 62,
      left: 72
    },
    xAxis: {
      type: "category",
      data: weatherModels.map((model) => model.meter),
      axisLabel: {
        interval: 0,
        rotate: weatherModels.length > 2 ? 20 : 0,
        width: 120,
        overflow: "break"
      }
    },
    yAxis: {
      type: "value"
    },
    series: [
      {
        name: "Actual",
        type: "bar",
        data: weatherModels.map((model) => model.actualConsumption),
        barMaxWidth: 34
      },
      {
        name: "Predicted",
        type: "bar",
        data: weatherModels.map((model) => model.predictedConsumption),
        barMaxWidth: 34
      },
      {
        name: "Savings",
        type: "line",
        data: weatherModels.map((model) => model.estimatedSavings),
        symbolSize: 8
      }
    ]
  };

  return (
    <div className="reportArtifactOverview">
      <div className="reportKpiStrip" aria-label="Report artifact indicators">
        <div className="reportMetric">
          <span>Client</span>
          <strong>{client}</strong>
        </div>
        <div className="reportMetric">
          <span>Site</span>
          <strong>{site}</strong>
        </div>
        <div className="reportMetric">
          <span>Source records</span>
          <strong>{formatNumber(totalRecords)}</strong>
        </div>
        <div className="reportMetric">
          <span>Source files</span>
          <strong>{formatNumber(sourcePackage.uploadCount)}</strong>
        </div>
        <div className="reportMetric">
          <span>Source types</span>
          <strong>{sourcePackage.categories.length > 0 ? sourcePackage.categories.join(", ") : "Not recorded"}</strong>
        </div>
        <div className="reportMetric">
          <span>Analysis input</span>
          <strong>{sourcePackage.executionInput}</strong>
        </div>
        {legacyBaseline ? (
          <>
            <div className="reportMetric">
              <span>Legacy meters</span>
              <strong>{formatNumber(legacyBaseline.meterCount)}</strong>
            </div>
            <div className="reportMetric">
              <span>Legacy rows</span>
              <strong>{formatNumber(legacyBaseline.energyRecords)}</strong>
            </div>
            <div className="reportMetric">
              <span>Weather rows</span>
              <strong>
                HDD {formatNumber(legacyBaseline.hddRecords)} / CDD {formatNumber(legacyBaseline.cddRecords)}
              </strong>
            </div>
            <div className="reportMetric">
              <span>Mapped meters</span>
              <strong>{formatNumber(legacyBaseline.mappedMeterCount)}</strong>
            </div>
            <div className="reportMetric">
              <span>Weather models</span>
              <strong>{formatNumber(legacyBaseline.weatherModelCount)}</strong>
            </div>
            <div className="reportMetric">
              <span>Normalized savings</span>
              <strong>{formatNumber(legacyBaseline.estimatedSavings)}</strong>
            </div>
            <div className="reportMetric">
              <span>Mapped SEUs</span>
              <strong>
                {legacyBaseline.seuCategories.length > 0
                  ? legacyBaseline.seuCategories.join(", ")
                  : "Not recorded"}
              </strong>
            </div>
          </>
        ) : null}
        <div className="reportMetric">
          <span>SEU consumption</span>
          <strong>{formatNumber(seuConsumption)}</strong>
        </div>
        <div className="reportMetric">
          <span>SEU per record</span>
          <strong>{formatNumber(intensity)}</strong>
        </div>
      </div>
      {metrics.length > 0 ? (
        <ReactECharts option={chartOption} className="reportArtifactChart" notMerge lazyUpdate />
      ) : (
        <div className="artifactEmptyState">
          <strong>No numeric artifact fields</strong>
          <span>The generated report can still be reviewed in the evidence table below.</span>
        </div>
      )}
      {weatherNormalization && (weatherModels.length > 0 || weatherNormalization.skippedMeters.length > 0) ? (
        <div className="weatherModelSection" aria-label="Weather-normalized model results">
          <div className="weatherModelHeader">
            <div>
              <strong>Weather-normalized models</strong>
              <span>
                {formatNumber(weatherNormalization.baselineYear)} baseline to{" "}
                {formatNumber(weatherNormalization.comparisonYear)} comparison
              </span>
            </div>
            <div>
              <strong>{formatSignedNumber(weatherNormalization.estimatedSavings)}</strong>
              <span>
                {formatNumber(weatherNormalization.modelCount)} modeled /{" "}
                {formatNumber(weatherNormalization.skippedMeterCount)} skipped
              </span>
            </div>
          </div>
          {weatherModels.length > 0 ? (
            <>
              <ReactECharts
                option={weatherModelChartOption}
                className="weatherModelChart"
                notMerge
                lazyUpdate
              />
              <div className="weatherModelTableWrap">
                <table className="weatherModelTable">
                  <thead>
                    <tr>
                      <th>Meter</th>
                      <th>SEU</th>
                      <th>Model</th>
                      <th>Actual</th>
                      <th>Predicted</th>
                      <th>Savings</th>
                      <th>R2</th>
                    </tr>
                  </thead>
                  <tbody>
                    {weatherModels.map((model) => (
                      <tr key={`${model.meter}-${model.normalization}`}>
                        <td>
                          <strong>{model.meter}</strong>
                          <span>{model.utility}</span>
                        </td>
                        <td>{model.seuCategory}</td>
                        <td>
                          <strong>{model.normalization.toUpperCase()}</strong>
                          <span>
                            {formatNumber(model.baselineRecords)} / {formatNumber(model.comparisonRecords)}
                          </span>
                        </td>
                        <td>{formatNumber(model.actualConsumption)}</td>
                        <td>{formatNumber(model.predictedConsumption)}</td>
                        <td>{formatSignedNumber(model.estimatedSavings)}</td>
                        <td>{formatNumber(model.r2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <div className="artifactEmptyState weatherModelEmpty">
              <strong>No weather-normalized models were fitted</strong>
              <span>Skipped meter diagnostics below show what blocked model creation.</span>
            </div>
          )}
          {weatherNormalization.skippedMeters.length > 0 ? (
            <div className="weatherModelTableWrap">
              <table className="weatherModelTable weatherSkipTable">
                <thead>
                  <tr>
                    <th>Skipped meter</th>
                    <th>SEU</th>
                    <th>Utility</th>
                    <th>Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {weatherNormalization.skippedMeters.map((skippedMeter) => (
                    <tr key={`${skippedMeter.meter}-${skippedMeter.reason}`}>
                      <td>
                        <strong>{skippedMeter.meter}</strong>
                      </td>
                      <td>{skippedMeter.seuCategory}</td>
                      <td>{skippedMeter.utility}</td>
                      <td>{labelFromKey(skippedMeter.reason)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
