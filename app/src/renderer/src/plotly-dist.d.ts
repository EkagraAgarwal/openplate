declare module 'plotly.js-dist-min' {
  interface PlotlyModule {
    newPlot: (
      el: HTMLElement,
      data: unknown[],
      layout?: unknown,
      config?: unknown
    ) => Promise<HTMLElement>
    react: (
      el: HTMLElement,
      data: unknown[],
      layout?: unknown,
      config?: unknown
    ) => Promise<HTMLElement>
    toImage: (
      el: HTMLElement,
      opts: { format: string; width?: number; height?: number; scale?: number }
    ) => Promise<string>
    purge: (el: HTMLElement) => void
  }
  const Plotly: PlotlyModule
  export default Plotly
}
