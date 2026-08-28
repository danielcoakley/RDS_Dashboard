"use client";

import ReactECharts from "echarts-for-react";

const option = {
  color: ["#12c2e0", "#f97316"],
  tooltip: {
    trigger: "axis"
  },
  legend: {
    top: 0,
    right: 0,
    data: ["Baseline", "Actual"]
  },
  grid: {
    top: 48,
    right: 24,
    bottom: 32,
    left: 48
  },
  xAxis: {
    type: "category",
    data: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
  },
  yAxis: {
    type: "value",
    axisLabel: {
      formatter: "{value} MWh"
    }
  },
  series: [
    {
      name: "Baseline",
      type: "line",
      smooth: true,
      data: [88, 84, 79, 76, 72, 69]
    },
    {
      name: "Actual",
      type: "bar",
      data: [82, 80, 73, 71, 68, 63]
    }
  ]
};

export function EnergyPerformanceChart() {
  return <ReactECharts option={option} className="energyChart" notMerge lazyUpdate />;
}
