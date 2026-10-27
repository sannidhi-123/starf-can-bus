import { Link, NavLink } from 'react-router-dom';
import type { ReactNode } from 'react';

type ConsoleNavKey = 'dashboard' | 'pipeline' | 'integration' | 'reports' | 'security';

type ConsoleLayoutProps = {
  activeNav: ConsoleNavKey;
  children: ReactNode;
};

function navItemClass(isActive: boolean) {
  if (isActive) {
    return 'flex items-center gap-3 px-3 py-2 bg-surface-container-lowest text-primary rounded-lg shadow-sm font-bold text-sm';
  }

  return 'flex items-center gap-3 px-3 py-2 text-on-surface-variant hover:bg-surface-container-high transition-transform duration-200 hover:translate-x-1 rounded-lg text-sm font-medium';
}

function topLinkClass(isActive: boolean) {
  if (isActive) {
    return 'text-primary font-bold border-b-2 border-primary pb-1';
  }

  return 'text-on-surface-variant hover:text-primary transition-colors';
}

export function ConsoleLayout({ activeNav, children }: ConsoleLayoutProps) {
  return (
    <div className="min-h-screen bg-surface font-body text-on-surface antialiased">
      <nav className="fixed top-0 z-50 w-full bg-surface-container-low/90 backdrop-blur-md">
        <div className="mx-auto flex h-16 w-full max-w-[1440px] items-center justify-between px-6">
          <div className="flex items-center gap-8">
            <Link to="/" className="font-headline text-xl font-bold tracking-tighter text-primary">
              STARF
            </Link>
            <div className="hidden gap-6 md:flex">
              <NavLink to="/" className={({ isActive }) => topLinkClass(activeNav !== 'security' && activeNav !== 'reports' && isActive)}>
                Home
              </NavLink>
              <NavLink to="/security" className={({ isActive }) => topLinkClass(activeNav === 'security' || isActive)}>
                About
              </NavLink>
              <NavLink to="/reports" className={({ isActive }) => topLinkClass(activeNav === 'reports' || isActive)}>
                Reports
              </NavLink>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button
              type="button"
              className="rounded-full p-2 text-on-surface-variant transition-all hover:bg-surface-container-highest"
              aria-label="Search"
            >
              <span className="material-symbols-outlined">search</span>
            </button>
            <button
              type="button"
              className="rounded-full p-2 text-on-surface-variant transition-all hover:bg-surface-container-highest"
              aria-label="Toggle theme"
            >
              <span className="material-symbols-outlined">dark_mode</span>
            </button>
            <Link
              to="/dashboard"
              className="rounded-xl bg-primary px-4 py-2 text-sm font-bold text-on-primary transition-transform active:scale-95"
            >
              Live Dashboard
            </Link>
          </div>
        </div>
      </nav>

      <aside className="fixed left-0 top-16 hidden h-[calc(100vh-64px)] w-64 flex-col gap-2 bg-surface-container-low p-4 lg:flex">
        <div className="mb-6 px-2">
          <h2 className="font-headline text-lg font-bold text-primary">STARF</h2>
          <p className="text-xs text-on-surface-variant opacity-70">Smart Traffic Anomaly Recognition Kit</p>
        </div>

        <nav className="flex-1 space-y-1">
          <NavLink to="/dashboard" className={({ isActive }) => navItemClass(activeNav === 'dashboard' || isActive)}>
            <span className="material-symbols-outlined">dashboard</span>
            Dashboard
          </NavLink>
          <NavLink to="/pipeline" className={({ isActive }) => navItemClass(activeNav === 'pipeline' || isActive)}>
            <span className="material-symbols-outlined">account_tree</span>
            Model &amp; Pipeline
          </NavLink>
          <NavLink to="/integration" className={({ isActive }) => navItemClass(activeNav === 'integration' || isActive)}>
            <span className="material-symbols-outlined">integration_instructions</span>
            API &amp; Integration
          </NavLink>
          <NavLink to="/reports" className={({ isActive }) => navItemClass(activeNav === 'reports' || isActive)}>
            <span className="material-symbols-outlined">analytics</span>
            Reports
          </NavLink>
          <NavLink to="/security" className={({ isActive }) => navItemClass(activeNav === 'security' || isActive)}>
            <span className="material-symbols-outlined">verified_user</span>
            Security
          </NavLink>
        </nav>

        <div className="mt-auto space-y-1">
          <button type="button" className="mb-4 w-full rounded-xl bg-primary py-2.5 text-sm font-bold text-on-primary shadow-lg shadow-primary/20">
            Deploy Model
          </button>
          <button
            type="button"
            className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm text-on-surface-variant transition-all hover:bg-surface-container-high"
          >
            <span className="material-symbols-outlined">settings</span>
            Settings
          </button>
          <button
            type="button"
            className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm text-on-surface-variant transition-all hover:bg-surface-container-high"
          >
            <span className="material-symbols-outlined">help</span>
            Support
          </button>
        </div>
      </aside>

      <main className="min-h-screen pt-16 lg:ml-64">{children}</main>

      <footer className="mt-auto bg-surface-container-low lg:ml-64">
        <div className="mx-auto flex w-full max-w-[1440px] flex-col items-center justify-between gap-8 px-8 py-12 md:flex-row">
          <div className="mb-6 md:mb-0">
            <span className="block font-headline text-lg font-bold text-primary">STARF</span>
            <p className="mt-2 text-xs text-on-surface-variant">© 2026 STARF — Explainable Hybrid Anomaly Detection System for Intra-Vehicular CAN-Bus Networks</p>
          </div>
          <div className="flex gap-8">
            <Link className="text-xs text-on-surface-variant transition-colors hover:text-primary" to="/security">
              Mission
            </Link>
            <Link className="text-xs text-on-surface-variant transition-colors hover:text-primary" to="/pipeline">
              Tech Stack
            </Link>
            <Link className="text-xs text-on-surface-variant transition-colors hover:text-primary" to="/integration">
              Documentation
            </Link>
            <Link className="text-xs text-on-surface-variant transition-colors hover:text-primary" to="/security">
              Contact
            </Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
