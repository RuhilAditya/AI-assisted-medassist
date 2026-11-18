// MedAssist mark: a plate label. A ring (the structure) and a leader line to its label dot.
export function Mark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect x="0.5" y="0.5" width="31" height="31" rx="7" fill="#14223b" />
      <circle cx="12" cy="18" r="6.2" fill="none" stroke="#ffffff" strokeWidth="2" />
      <circle cx="12" cy="18" r="1.9" fill="#c93a1b" />
      <path d="M16.4 13.6 L23.5 6.5" stroke="#c93a1b" strokeWidth="2" strokeLinecap="round" />
      <circle cx="24.5" cy="5.8" r="2.2" fill="#5d9ae0" />
    </svg>
  );
}
