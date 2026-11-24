import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { useAuth } from "./auth";
import { Shell } from "./components/shell";
import { Spinner } from "./components/ui";
import Login from "./pages/Login";
import Cases from "./pages/Cases";
import CaseDetail from "./pages/CaseDetail";
import ClipSetup from "./pages/ClipSetup";
import TransactionPage from "./pages/Transaction";
import TrackFaces from "./pages/TrackFaces";
import SearchPage from "./pages/Search";
import ExportPage from "./pages/Export";
import Admin from "./pages/Admin";
import Atms from "./pages/Atms";

function Protected({ children }: { children: React.ReactNode }) {
  const { user, ready } = useAuth();
  const loc = useLocation();
  if (!ready)
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner />
      </div>
    );
  if (!user) return <Navigate to="/login" state={{ from: loc.pathname }} replace />;
  return <Shell>{children}</Shell>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/cases" element={<Protected><Cases /></Protected>} />
      <Route path="/cases/:id" element={<Protected><CaseDetail /></Protected>} />
      <Route path="/cases/:id/export" element={<Protected><ExportPage /></Protected>} />
      <Route path="/clips/:id/setup" element={<Protected><ClipSetup /></Protected>} />
      <Route path="/transactions/:id" element={<Protected><TransactionPage /></Protected>} />
      <Route path="/tracks/:id/faces" element={<Protected><TrackFaces /></Protected>} />
      <Route path="/search/:id" element={<Protected><SearchPage /></Protected>} />
      <Route path="/atms" element={<Protected><Atms /></Protected>} />
      <Route path="/admin" element={<Protected><Admin /></Protected>} />
      <Route path="*" element={<Navigate to="/cases" replace />} />
    </Routes>
  );
}
