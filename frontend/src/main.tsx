import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";
import "./target-explorer.css";
import "./aws-branding.css";
import "./target-state.css";

createRoot(document.getElementById("root")!).render(<StrictMode><App /></StrictMode>);
