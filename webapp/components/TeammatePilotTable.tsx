import {
  TEAMMATE_PILOT,
  TEAMMATE_PILOT_CAVEAT,
  TEAMMATE_PILOT_FINDING,
  TEAMMATE_PILOT_FULL_SCORE,
  TEAMMATE_PILOT_METRIC,
} from "@/lib/copy";

export function TeammatePilotTable() {
  return (
    <div>
      <p className="mb-2 text-[11px] text-text-faint">Metric: {TEAMMATE_PILOT_METRIC}</p>
      <table className="w-full text-left text-[12px]">
        <thead>
          <tr className="text-[10px] uppercase tracking-wide text-text-faint">
            <th className="pb-2 pr-3 font-medium">Method</th>
            <th className="pb-2 pr-3 font-medium">10% cache retained</th>
            <th className="pb-2 font-medium">20% cache retained</th>
          </tr>
        </thead>
        <tbody>
          {TEAMMATE_PILOT.map((row) => (
            <tr key={row.label} className="border-t border-border-soft">
              <td className="py-1.5 pr-3 font-semibold text-text">{row.label}</td>
              <td className="py-1.5 pr-3 text-text-dim">{row.pct10}%</td>
              <td className="py-1.5 text-text-dim">{row.pct20}%</td>
            </tr>
          ))}
          <tr className="border-t border-border-soft">
            <td className="py-1.5 pr-3 font-semibold text-text">Full (uncompressed)</td>
            <td className="py-1.5 pr-3 text-text-dim" colSpan={2}>
              {TEAMMATE_PILOT_FULL_SCORE}%
            </td>
          </tr>
        </tbody>
      </table>
      <p className="mt-3 text-[12px] leading-relaxed text-text">{TEAMMATE_PILOT_FINDING}</p>
      <p className="mt-1.5 text-[10.5px] leading-relaxed text-text-faint">{TEAMMATE_PILOT_CAVEAT}</p>
    </div>
  );
}
