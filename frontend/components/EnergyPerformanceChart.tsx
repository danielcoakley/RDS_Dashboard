"use client";

import ReactECharts from "echarts-for-react";

export type SourceReadinessPoint = {
  label: string;
  ready: boolean;
  storedCount: number;
};

export type StatusCountPoint = {
  status: string;
  count: number;
};

export type MeterMixPoint = {
  commodity: string;
  total: number;
  seu: number;
};

type EnergyPerformanceChartProps = {
  sourceReadiness: SourceReadinessPoint[];
  runStatuses: StatusCountPoint[];
  jobStatuses: StatusCountPoint[];
  meterMix: MeterMixPoint[];
};

const statusColors: Record<string, string> = {
  queued: "#64748b",
  running: "#12c2e0",
  succeeded: "#0f9cae",
  failed: "#f97316"
};

const lifecycleStatuses = ["queued", "running", "succeeded", "failed"];

type CountLabelParams = {
  value?: number | string;
};

function statusValue(points: StatusCountPoint[], status: string): number {
  return points.find((point) => point.status === status)?.count ?? 0;
}

function visibleCountLabel(params: CountLabelParams): string {
  const value = Number(params.value ?? 0);
  return value > 0 ? String(value) : "";
}

export function EnergyPerformanceChart({
  sourceReadiness,
  runStatuses,
  jobStatuses,
  meterMix
}: EnergyPerformanceChartProps) {
  const sourceLabels = sourceReadiness.map((point) => point.label);
  const commodityLabels = meterMix.map((point) => point.commodity);
  const lifecycleTotals = [
    runStatuses.reduce((total, point) => total + point.count, 0),
    jobStatuses.reduce((total, point) => total + point.count, 0)
  ];
  const sourceChart = {
    color: ["#0f9cae", "#dbe3ef"],
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" }
    },
    grid: {
      top: 24,
      right: 18,
      bottom: 32,
      left: 102
    },
    xAxis: {
      type: "value",
      minInterval: 1,
      max: Math.max(1, ...sourceReadiness.map((point) => point.storedCount))
    },
    yAxis: {
      type: "category",
      data: sourceLabels,
      axisLabel: {
        interval: 0,
        width: 82,
        overflow: "break"
      }
    },
    series: [
      {
        name: "Stored files",
        type: "bar",
        data: sourceReadiness.map((point) => ({
          value: point.storedCount,
          itemStyle: {
            color: point.ready ? "#0f9cae" : "#dbe3ef"
          }
        })),
        barMaxWidth: 24,
        label: {
          show: true,
          position: "right",
          formatter: visibleCountLabel
        }
      }
    ]
  };

  const lifecycleChart = {
    color: lifecycleStatuses.map((status) => statusColors[status]),
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" }
    },
    legend: {
      top: 0,
      left: "center"
    },
    grid: {
      top: 58,
      right: 18,
      bottom: 24,
      left: 54
    },
    xAxis: {
      type: "value",
      minInterval: 1,
      max: Math.max(1, ...lifecycleTotals)
    },
    yAxis: {
      type: "category",
      data: ["Runs", "Jobs"]
    },
    series: lifecycleStatuses.map((status) => ({
      name: status,
      type: "bar",
      stack: "lifecycle",
      emphasis: {
        focus: "series"
      },
      data: [statusValue(runStatuses, status), statusValue(jobStatuses, status)],
      label: {
        show: true,
        formatter: visibleCountLabel
      }
    }))
  };

  const meterChart = {
    color: ["#12c2e0", "#f97316"],
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" }
    },
    legend: {
      top: 0,
      right: 0,
      data: ["All meters", "SEU meters"]
    },
    grid: {
      top: 44,
      right: 18,
      bottom: 44,
      left: 42
    },
    xAxis: {
      type: "category",
      data: commodityLabels.length ? commodityLabels : ["No meters"],
      axisLabel: {
        interval: 0,
        width: 72,
        overflow: "truncate"
      }
    },
    yAxis: {
      type: "value",
      minInterval: 1
    },
    series: [
      {
        name: "All meters",
        type: "bar",
        data: meterMix.length ? meterMix.map((point) => point.total) : [0],
        barMaxWidth: 34
      },
      {
        name: "SEU meters",
        type: "bar",
        data: meterMix.length ? meterMix.map((point) => point.seu) : [0],
        barMaxWidth: 34
      }
    ]
  };

  return (
    <div className="tenantCharts" aria-label="Tenant workflow charts">
      <div className="tenantChart">
        <div className="tenantChartHeader">
          <strong>Source readiness</strong>
          <span>Stored files by required category</span>
        </div>
        <ReactECharts option={sourceChart} className="energyChart" notMerge lazyUpdate />
      </div>
      <div className="tenantChart">
        <div className="tenantChartHeader">
          <strong>Run lifecycle</strong>
          <span>Runs and queued jobs by status</span>
        </div>
        <ReactECharts option={lifecycleChart} className="energyChart" notMerge lazyUpdate />
      </div>
      <div className="tenantChart">
        <div className="tenantChartHeader">
          <strong>Meter mix</strong>
          <span>Configured meters and SEUs</span>
        </div>
        <ReactECharts option={meterChart} className="energyChart" notMerge lazyUpdate />
      </div>
    </div>
  );
}
