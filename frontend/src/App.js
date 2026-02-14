import React, { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Activity, Camera, Users, Brain, 
  AlertCircle, CheckCircle, Zap 
} from "lucide-react";
import {
  Box, Typography, Grid, Chip, LinearProgress
} from "@mui/material";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer
} from "recharts";

const GlassCard = ({ children, title, icon: Icon }) => (
  <motion.div 
    initial={{ opacity: 0, y: 20 }} 
    animate={{ opacity: 1, y: 0 }}
    style={{
      background: "rgba(255, 255, 255, 0.05)",
      backdropFilter: "blur(16px)",
      borderRadius: "24px",
      border: "1px solid rgba(255, 255, 255, 0.1)",
      padding: "20px",
      height: "100%",
      boxShadow: "0 8px 32px 0 rgba(0, 0, 0, 0.37)",
    }}
  >
    <Box display="flex" alignItems="center" mb={2}>
      {Icon && <Icon size={20} style={{ color: "#00e5ff", marginRight: "10px" }} />}
      <Typography variant="subtitle1" sx={{ color: "rgba(255,255,255,0.7)", fontWeight: 600 }}>
        {title}
      </Typography>
    </Box>
    {children}
  </motion.div>
);

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [data, setData] = useState({ state: "Waiting", engagement: 0, coords: { x: 0.5, y: 0.5 } });
  const [chartData, setChartData] = useState([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => { if (videoRef.current) videoRef.current.srcObject = stream; });

    const ws = new WebSocket("wss://ai-engagement-dashboard.onrender.com/ws");
    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (event) => {
      const res = JSON.parse(event.data);
      setData(res);
      setChartData(prev => [...prev.slice(-30), { time: Date.now(), val: res.engagement }]);
    };

    const interval = setInterval(() => {
      if (ws.readyState === 1 && videoRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoRef.current, 0, 0, 320, 240);
        ws.send(canvas.toDataURL("image/jpeg", 0.5));
      }
    }, 200);

    return () => { clearInterval(interval); ws.close(); };
  }, []);

  const getStatusColor = () => {
    if (data.state === "Focused") return "#00ff99";
    if (data.state === "Distracted") return "#ffcc00";
    if (data.state === "Drowsy") return "#ff4444";
    return "#aaa";
  };

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: "#0a0e14", p: 4, color: "white", fontFamily: "'Inter', sans-serif" }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Box>
          <Typography variant="h4" fontWeight="800" sx={{ letterSpacing: "-1px" }}>
            NEURAL<span style={{ color: "#00e5ff" }}>SIGHT</span>
          </Typography>
          <Typography variant="caption" sx={{ opacity: 0.5 }}>AI-POWERED ENGAGEMENT ANALYTICS v2.4</Typography>
        </Box>
        <Chip 
          icon={<Activity size={14} color={connected ? "#00ff99" : "#ff4444"} />}
          label={connected ? "SYSTEM LIVE" : "OFFLINE"} 
          sx={{ bgcolor: "rgba(255,255,255,0.05)", color: "white", border: "1px solid rgba(255,255,255,0.1)" }}
        />
      </Box>

      <Grid container spacing={3}>
        {/* Main Camera View */}
        <Grid item xs={12} md={7}>
          <GlassCard title="Real-time Neural Feed" icon={Camera}>
            <Box position="relative" sx={{ borderRadius: "16px", overflow: "hidden", bgcolor: "black" }}>
              <video ref={videoRef} autoPlay width="100%" style={{ display: "block" }} />
              
              {/* Dynamic Face Tracker Overlay */}
              <motion.div
                animate={{ left: `${data.coords.x * 100}%`, top: `${data.coords.y * 100}%` }}
                style={{
                  position: "absolute",
                  width: "120px",
                  height: "120px",
                  border: `2px solid ${getStatusColor()}`,
                  borderRadius: "50%",
                  transform: "translate(-50%, -50%)",
                  boxShadow: `0 0 20px ${getStatusColor()}`,
                  pointerEvents: "none"
                }}
              >
                <Box sx={{ position: "absolute", top: -25, left: "50%", transform: "translateX(-50%)", bgcolor: getStatusColor(), color: "black", px: 1, borderRadius: "4px", fontSize: "10px", fontWeight: "bold" }}>
                  {data.state.toUpperCase()}
                </Box>
              </motion.div>
            </Box>
          </GlassCard>
        </Grid>

        {/* Analytics Sidebar */}
        <Grid item xs={12} md={5}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <GlassCard title="Engagement Index" icon={Brain}>
                <Typography variant="h2" fontWeight="800" color="#00e5ff">
                  {(data.engagement * 100).toFixed(0)}%
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={data.engagement * 100} 
                  sx={{ height: 8, borderRadius: 4, mt: 2, bgcolor: "rgba(255,255,255,0.1)", "& .MuiLinearProgress-bar": { bgcolor: "#00e5ff" } }} 
                />
              </GlassCard>
            </Grid>

            <Grid item xs={12}>
              <GlassCard title="Biometric Trend" icon={Zap}>
                <Box height={160}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData}>
                      <defs>
                        <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00e5ff" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#00e5ff" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <Area type="monotone" dataKey="val" stroke="#00e5ff" fillOpacity={1} fill="url(#colorVal)" strokeWidth={3} isAnimationActive={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </Box>
              </GlassCard>
            </Grid>
          </Grid>
        </Grid>
      </Grid>

      <canvas ref={canvasRef} width="320" height="240" style={{ display: "none" }} />
    </Box>
  );
}

export default App;
