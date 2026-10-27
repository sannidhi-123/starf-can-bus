// FILE: frontend/src/layout/AppLayout.tsx - BUGS FIXED: 2, 4
import { createContext, useContext, useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { Activity, FileCheck2, HelpCircle, Play, ShieldCheck, Square } from 'lucide-react';
import { apiClient, type HealthResponse } from '../api/client';

type AppLayoutProps = {
  children?: ReactNode;
};

type AnalysisControlContextValue = {
  isAnalysisRunning: boolean;
  setIsAnalysisRunning: (next: boolean) => void;
  toggleAnalysis: () => void;
};

const AnalysisControlContext = createContext<AnalysisControlContextValue | null>(null);

const NAV_ITEMS = [
  { to: '/dashboard', label: 'Dashboard', icon: Activity },
  { to: '/integration', label: 'Analyze', icon: ShieldCheck },
  { to: '/reports', label: 'Reports', icon: FileCheck2 },
  { to: '/security', label: 'About', icon: HelpCircle },
];

const OUTER_STYLE: CSSProperties = {
  display: 'flex',
  height: '100vh',
  width: '100vw',
  overflow: 'hidden',
  background: 'var(--color-bg-base)',
};

const SIDEBAR_STYLE: CSSProperties = {
  width: '220px',
  minWidth: '220px',
  maxWidth: '220px',
  height: '100vh',
  display: 'flex',
  flexDirection: 'column',
  flexShrink: 0,
  background: 'var(--color-bg-sidebar)',
  borderRight: '1px solid var(--color-border)',
  zIndex: 10,
  overflowY: 'auto',
  overflowX: 'hidden',
  boxShadow: 'var(--shadow-md)',
};

const MAIN_STYLE: CSSProperties = {
  flex: 1,
  height: '100vh',
  overflowY: 'auto',
  overflowX: 'hidden',
  display: 'flex',
  flexDirection: 'column',
  minWidth: 0,
};

const OUTLET_WRAPPER_STYLE: CSSProperties = {
  flex: 1,
  padding: '24px',
  maxWidth: '100%',
  boxSizing: 'border-box',
};

function getNavLinkStyle(isActive: boolean): CSSProperties {
  return {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '9px 12px',
    borderRadius: '7px',
    fontSize: '0.83rem',
    fontWeight: isActive ? 600 : 500,
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    width: '100%',
    boxSizing: 'border-box',
    textDecoration: 'none',
    cursor: 'pointer',
    color: isActive ? 'var(--color-primary)' : 'var(--color-text-secondary)',
    background: isActive ? 'var(--color-primary-light)' : 'transparent',
    transition: 'all 0.15s ease',
  };
}

export function useAnalysisControl(): AnalysisControlContextValue {
  const context = useContext(AnalysisControlContext);
  if (!context) {
    throw new Error('useAnalysisControl must be used within AppLayout');
  }
  return context;
}

export function AppLayout({ children }: AppLayoutProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isAnalysisRunning, setIsAnalysisRunning] = useState(true);
  const location = useLocation();

  useEffect(() => {
    let active = true;

    const checkHealth = () => {
      apiClient
        .getHealth()
        .then((data) => {
          if (active) {
            setHealth(data);
          }
        })
        .catch(() => {
          if (active) {
            setHealth(null);
          }
        });
    };

    checkHealth();
    const interval = setInterval(checkHealth, 4000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [location.pathname]);

  const pageLabel = useMemo(() => {
    const entry = NAV_ITEMS.find((item) => location.pathname.startsWith(item.to));
    return entry?.label ?? 'Dashboard';
  }, [location.pathname]);

  const contextValue = useMemo(
    () => ({
      isAnalysisRunning,
      setIsAnalysisRunning,
      toggleAnalysis: () => setIsAnalysisRunning((prev) => !prev),
    }),
    [isAnalysisRunning]
  );

  const ToggleIcon = isAnalysisRunning ? Square : Play;

  return (
    <AnalysisControlContext.Provider value={contextValue}>
      <div style={OUTER_STYLE}>
        <aside style={SIDEBAR_STYLE}>
          <div
            style={{
              height: '56px',
              padding: '0 20px',
              borderBottom: '1px solid var(--color-border)',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <Link
              to="/dashboard"
              style={{
                textDecoration: 'none',
                display: 'flex',
                flexDirection: 'column',
                lineHeight: 1,
              }}
            >
              <span
                style={{
                  fontSize: '1rem',
                  fontWeight: 800,
                  color: 'var(--color-primary)',
                  letterSpacing: '-0.02em',
                }}
              >
              STARF
              </span>
              <span
                style={{
                  marginTop: '4px',
                  fontSize: '0.65rem',
                  color: 'var(--color-text-muted)',
                  letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                }}
              >
                Smart Traffic Anomaly Recognition Kit
              </span>
            </Link>
          </div>

          <div style={{ padding: '16px 12px', flexGrow: 1 }}>
            <div
              style={{
                fontSize: '0.62rem',
                fontWeight: 600,
                color: 'var(--color-text-muted)',
                letterSpacing: '0.1em',
                padding: '0 8px',
                marginBottom: '6px',
                textTransform: 'uppercase',
              }}
            >
              Navigation
            </div>

            <nav style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink key={item.to} to={item.to} style={({ isActive }) => getNavLinkStyle(isActive)} className="hover:bg-bg-hover">
                    <Icon size={16} />
                    <span
                      style={{
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        flex: 1,
                        minWidth: 0,
                      }}
                    >
                      {item.label}
                    </span>
                  </NavLink>
                );
              })}
            </nav>
          </div>

          <div style={{ padding: '12px', borderTop: '1px solid var(--color-border)' }}>
            <button
              type="button"
              onClick={contextValue.toggleAnalysis}
              className={isAnalysisRunning ? 'hover:bg-[#ffe6ec] hover:shadow-sm' : 'hover:bg-[#e0f0ff] hover:shadow-sm'}
              style={{
                background: isAnalysisRunning ? 'var(--color-danger-bg)' : 'var(--color-primary-light)',
                color: isAnalysisRunning ? 'var(--color-danger)' : 'var(--color-primary)',
                border: isAnalysisRunning ? '1px solid rgba(196, 30, 58, 0.2)' : '1px solid rgba(26, 111, 212, 0.2)',
                borderRadius: '8px',
                padding: '10px 16px',
                fontSize: '0.8rem',
                fontWeight: 600,
                width: '100%',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'all 0.15s ease',
              }}
            >
              <ToggleIcon size={14} />
              <span>{isAnalysisRunning ? 'Stop Analysis' : 'Start Analysis'}</span>
            </button>

            <div
              style={{
                marginTop: '10px',
                display: 'flex',
                alignItems: 'center',
                gap: '7px',
                fontSize: '0.72rem',
                color: 'var(--color-text-muted)',
              }}
            >
              <span
                style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '999px',
                  background: isAnalysisRunning ? 'var(--color-success)' : '#94a3b8',
                }}
              />
              <span>{isAnalysisRunning ? 'System Online' : 'System Paused'}</span>
            </div>
          </div>
        </aside>

        <main style={MAIN_STYLE}>
          <header
            style={{
              height: '56px',
              borderBottom: '1px solid var(--color-border)',
              boxShadow: 'var(--shadow-xs)',
              background: 'var(--color-bg-card)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 20px',
              position: 'sticky',
              top: 0,
              zIndex: 5,
            }}
          >
            <div>
              <h1
                style={{
                  fontSize: '1.1rem',
                  fontWeight: 700,
                  color: 'var(--color-text-primary)',
                  letterSpacing: '-0.01em',
                  lineHeight: 1.2,
                }}
              >
                {pageLabel}
              </h1>
              <p style={{ fontSize: '0.72rem', color: 'var(--color-text-muted)' }}>Smart Traffic Anomaly Recognition Kit</p>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.72rem', color: 'var(--color-text-muted)' }}>
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '5px',
                  fontWeight: 600,
                  padding: '2px 8px',
                  borderRadius: '999px',
                  fontSize: '0.65rem',
                  letterSpacing: '0.05em',
                  textTransform: 'uppercase',
                  background: health ? 'rgba(15,123,85,0.12)' : 'rgba(196,30,58,0.12)',
                  color: health ? 'var(--color-success)' : 'var(--color-danger)',
                  border: health ? '1px solid rgba(15,123,85,0.25)' : '1px solid rgba(196,30,58,0.25)',
                }}
              >
                <span
                  style={{
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    background: health ? 'var(--color-success)' : 'var(--color-danger)',
                  }}
                />
                {health ? 'ONLINE' : 'OFFLINE'}
              </span>
              <span>Model: {health ? (health.model || 'Random Forest + rules') : 'unknown'}</span>
              <span>|</span>
              <span>Device: {health?.device ?? 'n/a'}</span>
            </div>
          </header>

          <div style={OUTLET_WRAPPER_STYLE}>{children ?? <Outlet />}</div>
        </main>
      </div>
    </AnalysisControlContext.Provider>
  );
}