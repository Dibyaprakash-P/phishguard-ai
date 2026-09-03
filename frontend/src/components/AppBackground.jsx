/**
 * Full-viewport background wrapper.
 *
 * Layer order (bottom to top):
 *   1. the supplied artwork, fixed and cover-sized so it never repeats or
 *      shifts while the page scrolls;
 *   2. a vertical scrim that darkens the top and bottom edges just enough for
 *      text contrast while leaving the bright centre of the artwork visible;
 *   3. a soft accent bloom that ties the UI's blue to the art;
 *   4. the application content.
 *
 * `background-attachment: fixed` is unreliable on iOS Safari, so the artwork
 * is a fixed-position element instead of a body background.
 */
export default function AppBackground({ children }) {
  return (
    <div className="relative min-h-screen">
      {/* 1 — artwork */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 -z-30 bg-ink-900 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: "url('/assets/image.png')" }}
      />

      {/* 2 — readability scrim */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 -z-20"
        style={{
          background:
            'linear-gradient(180deg, rgba(3,6,15,0.82) 0%, rgba(3,6,15,0.42) 26%, rgba(3,6,15,0.34) 55%, rgba(3,6,15,0.8) 100%)',
        }}
      />

      {/* 3 — accent bloom */}
      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 -z-10"
        style={{
          background:
            'radial-gradient(1100px 620px at 50% 14%, rgba(44,192,255,0.1) 0%, transparent 62%)',
        }}
      />

      {/* 4 — content */}
      <div className="relative z-0">{children}</div>
    </div>
  )
}
