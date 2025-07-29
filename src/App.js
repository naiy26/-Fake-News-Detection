import React, { useState, useEffect } from "react";
import * as THREE from "three";
import {
  AlertTriangle,
  Check,
  AlertCircle,
  Droplet,
  Search,
  Info,
  FileText,
  User,
  Calendar,
  Globe,
  Link,
  BarChart2,
  TrendingUp,
  Bookmark,
  Shield,
  Coffee,
  ChevronRight,
  Clock,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from "recharts";

const TruthLensApp = () => {
  const [claim, setClaim] = useState("");
  const [useRag, setUseRag] = useState(true);
  const [useKg, setUseKg] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("analysis");
  const [demoSamples, setDemoSamples] = useState([]);
  const [apiBaseUrl, setApiBaseUrl] = useState("http://localhost:8000");
  const [fakeNewsStats, setFakeNewsStats] = useState({});
  const [categoryBreakdown, setCategoryBreakdown] = useState([]);
  const [credibilitySourceData, setCredibilitySourceData] = useState([]);
  const [fakeNewsExamples, setFakeNewsExamples] = useState([]);
  const [activeDashboardTab, setActiveDashboardTab] = useState("trends");
  // Add these state variables
  const [promptSuggestions, setPromptSuggestions] = useState(null);
  const [showPromptSuggestions, setShowPromptSuggestions] = useState(false);
  const [improvedPrompt, setImprovedPrompt] = useState("");
  const [promptAnalysisTimeout, setPromptAnalysisTimeout] = useState(null);
  // Add with your other state variables
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showResultsAnimation, setShowResultsAnimation] = useState(false);

  const [showShareModal, setShowShareModal] = useState(false);
  const COLORS = [
    "#0088FE",
    "#00C49F",
    "#FFBB28",
    "#FF8042",
    "#8884d8",
    "#82ca9d",
  ];
  const [isPageLoading, setIsPageLoading] = useState(true);

  React.useEffect(() => {
    return () => {
      // Cancel any pending prompt analysis requests when component unmounts
      if (window.lastPromptController) {
        window.lastPromptController.abort();
      }

      // Clear any pending timeouts
      if (promptAnalysisTimeout) {
        clearTimeout(promptAnalysisTimeout);
      }
    };
  }, [promptAnalysisTimeout]);

  // Add this useEffect for the loading animation
  useEffect(() => {
    // Simulate loading time
    const timer = setTimeout(() => {
      setIsPageLoading(false);
    }, 1500); // 1.5 seconds loading time

    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    console.log("Result updated:", result);
    if (result && result.ai_detection) {
      console.log("AI Detection data:", result.ai_detection);
    } else if (result) {
      console.log("Result exists but no AI detection data found");
    }
  }, [result]);

  useEffect(() => {
    if (result) {
      console.log("Analysis result:", result);
      console.log("Confidence value:", result.confidence);
      console.log("Verdict:", result.verdict);
    }
  }, [result]);

  useEffect(() => {
    // Mock statistics data
    const monthlyStats = [
      { month: "Jan", fakeNews: 423, misleading: 312, credible: 892 },
      { month: "Feb", fakeNews: 456, misleading: 289, credible: 921 },
      { month: "Mar", fakeNews: 521, misleading: 345, credible: 875 },
      { month: "Apr", fakeNews: 489, misleading: 367, credible: 843 },
      { month: "May", fakeNews: 468, misleading: 378, credible: 867 },
      { month: "Jun", fakeNews: 512, misleading: 399, credible: 834 },
      { month: "Jul", fakeNews: 567, misleading: 412, credible: 812 },
      { month: "Aug", fakeNews: 589, misleading: 423, credible: 798 },
      { month: "Sep", fakeNews: 612, misleading: 445, credible: 785 },
      { month: "Oct", fakeNews: 634, misleading: 467, credible: 768 },
      { month: "Nov", fakeNews: 612, misleading: 432, credible: 789 },
      { month: "Dec", fakeNews: 587, misleading: 401, credible: 823 },
    ];
    setFakeNewsStats(monthlyStats);

    // Category breakdown data
    const categories = [
      { name: "Health", value: 32 },
      { name: "Politics", value: 28 },
      { name: "Science", value: 12 },
      { name: "Celebrity", value: 15 },
      { name: "Finance", value: 8 },
      { name: "Other", value: 5 },
    ];
    setCategoryBreakdown(categories);

    // Credibility source data
    const credSources = [
      { name: "Unknown Sources", value: 65 },
      { name: "Social Media", value: 20 },
      { name: "Blogs", value: 10 },
      { name: "Mainstream Media", value: 5 },
    ];
    setCredibilitySourceData(credSources);

    // Fake news examples with images
    const examples = [
      {
        title: "False Claim: Moon Landing Was Faked",
        description:
          "This conspiracy theory claims that NASA's Apollo 11 mission was filmed in a studio. Multiple lines of evidence disprove this, including independent verification from multiple countries and the physical evidence left on the moon.",
        image: "/api/placeholder/500/300",
        source: "conspiracy-blog.example.com",
        date: "March 15, 2024",
        author: "Anonymous Writer",
        credibilityScore: "12/100",
      },
      {
        title: "Misleading: New Medication Cures All Diseases",
        description:
          "A widely shared article claimed that scientists developed a pill that cures all diseases. The actual research only showed promising results for treating a specific condition in early animal trials.",
        image: "/api/placeholder/500/300",
        source: "health-revolution.example.net",
        date: "June 8, 2024",
        author: "Health Guru",
        credibilityScore: "23/100",
      },
      {
        title: "Fabricated: Politician's Controversial Statement",
        description:
          "A viral post attributed an inflammatory quote to a prominent politician. Investigation found the quote was entirely fabricated and no record exists of the politician ever making such a statement.",
        image: "/api/placeholder/500/300",
        source: "political-truths.example.org",
        date: "August 23, 2024",
        author: "Political Insider",
        credibilityScore: "8/100",
      },
    ];
    setFakeNewsExamples(examples);
    // Load demo samples
    const samples = [
      {
        title: "COVID-19 Conspiracy",
        claim: "COVID-19 vaccines contain microchips for tracking people.",
      },
      {
        title: "Climate Change Denial",
        claim:
          "Scientific studies prove that climate change is a hoax invented to get more funding.",
      },
      {
        title: "Miracle Cure",
        claim:
          "Doctors discovered that drinking hot water with lemon cures all types of cancer.",
      },
    ];
    setDemoSamples(samples);
  }, []);

  // Add this component to your file
  const ParallaxHeader = () => {
    const [offset, setOffset] = React.useState(0);

    React.useEffect(() => {
      const handleScroll = () => {
        setOffset(window.pageYOffset);
      };

      window.addEventListener("scroll", handleScroll);

      return () => {
        window.removeEventListener("scroll", handleScroll);
      };
    }, []);

    return (
      <div className="relative overflow-hidden bg-gradient-to-r from-blue-600 to-indigo-700">
        {/* Parallax background elements */}

        <div className="absolute inset-0 z-0 overflow-hidden">
          <WaterWaveEffect />
        </div>

        <div
          className="absolute inset-0 opacity-10"
          style={{
            transform: `translateY(${offset * 0.3}px)`,
            background: `radial-gradient(circle at 30% 50%, rgba(255,255,255,0.2) 0%, rgba(0,0,0,0) 50%),
                      radial-gradient(circle at 70% 80%, rgba(255,255,255,0.1) 0%, rgba(0,0,0,0) 40%)`,
          }}
        ></div>

        {/* Floating particles */}
        <div className="absolute inset-0">
          <div
            className="absolute top-1/4 left-1/4 w-12 h-12 bg-white rounded-full opacity-10"
            style={{
              transform: `translate(${-offset * 0.1}px, ${offset * 0.05}px)`,
            }}
          ></div>
          <div
            className="absolute top-3/4 left-2/3 w-20 h-20 bg-white rounded-full opacity-5"
            style={{
              transform: `translate(${offset * 0.15}px, ${-offset * 0.1}px)`,
            }}
          ></div>
          <div
            className="absolute top-1/2 left-3/4 w-8 h-8 bg-white rounded-full opacity-10"
            style={{
              transform: `translate(${-offset * 0.2}px, ${-offset * 0.07}px)`,
            }}
          ></div>
        </div>

        {/* Header content */}
        <div className="container mx-auto px-4 py-6 relative z-10">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <Droplet className="text-white mr-2" size={28} />
              <h1 className="text-2xl font-bold text-white">TruthLens</h1>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => setShowHistory(true)}
                className="text-white text-sm flex items-center transition-transform hover:scale-105"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 mr-1"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                History
              </button>
              <div className="text-white text-sm md:text-base flex items-center">
                <Shield className="mr-2" size={18} />
                Advanced Fake News Detection System
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const ResultsRevealAnimation = ({ result, isVisible, onComplete }) => {
    const [showAnimation, setShowAnimation] = React.useState(isVisible);
    const [animationComplete, setAnimationComplete] = React.useState(false);

    React.useEffect(() => {
      if (isVisible) {
        setShowAnimation(true);

        // Trigger completion after animation
        const timer = setTimeout(() => {
          setAnimationComplete(true);
          onComplete();
        }, 2500);

        return () => clearTimeout(timer);
      }
    }, [isVisible, onComplete]);

    if (!showAnimation) return null;

    const getResultColor = () => {
      if (!result || !result.verdict) return "#9ca3af";

      const verdict = result.verdict.toLowerCase();
      if (verdict.includes("true") && !verdict.includes("partially")) {
        return "#10b981"; // Green
      } else if (
        verdict.includes("partially") ||
        verdict.includes("misleading")
      ) {
        return "#f59e0b"; // Yellow
      } else if (verdict.includes("false") || verdict.includes("fake")) {
        return "#ef4444"; // Red
      }
      return "#9ca3af"; // Gray default
    };

    const resultColor = getResultColor();
    const confidence = result?.confidence || 0;

    return (
      <div
        className={`fixed inset-0 flex items-center justify-center bg-white z-50 ${
          animationComplete ? "animate-fade-out" : ""
        }`}
      >
        <div className="text-center px-6 py-8 max-w-md">
          <div className="relative mb-8 mx-auto">
            {/* Animated circles */}
            <div className="w-40 h-40 rounded-full border-8 border-gray-100 flex items-center justify-center relative mx-auto">
              <svg
                className="w-full h-full absolute top-0 left-0"
                viewBox="0 0 100 100"
              >
                <circle
                  cx="50"
                  cy="50"
                  r="46"
                  fill="none"
                  stroke="#e5e7eb"
                  strokeWidth="8"
                />
                <circle
                  cx="50"
                  cy="50"
                  r="46"
                  fill="none"
                  stroke={resultColor}
                  strokeWidth="8"
                  strokeDasharray={`${confidence * 2.89} 289`}
                  strokeDashoffset="0"
                  transform="rotate(-90 50 50)"
                  className="animate-fill-progress"
                  style={{
                    strokeDasharray: `${confidence * 2.89} 289`,
                    transition: "stroke-dasharray 2s ease-in-out",
                  }}
                />
              </svg>
              <div
                className="text-3xl font-bold"
                style={{ color: resultColor }}
              >
                {confidence}%
              </div>
            </div>

            {/* Verdict text with typing animation */}
            <div className="h-10 flex items-center justify-center overflow-hidden">
              <div
                className="text-2xl font-bold animate-typing-effect"
                style={{ color: resultColor }}
              >
                {result?.verdict || "Analysis Complete"}
              </div>
            </div>
          </div>

          <div
            className="mt-4 animate-fade-in"
            style={{ animationDelay: "1s" }}
          >
            <p className="text-gray-700">
              {result?.explanation?.substring(0, 100) ||
                "Your analysis is ready to view."}
              {result?.explanation?.length > 100 ? "..." : ""}
            </p>
          </div>
        </div>
      </div>
    );
  };

  // Add this to your TruthLensApp component
  const EnhancedLoadingAnimation = () => {
    return (
      <div className="fixed inset-0 flex items-center justify-center bg-white z-50">
        <div className="text-center">
          <div className="relative w-24 h-24 mx-auto mb-4">
            {/* Animated rings */}
            <div className="absolute inset-0 rounded-full border-4 border-blue-200 opacity-20"></div>
            <div className="absolute inset-0 rounded-full border-4 border-blue-400 opacity-30 animate-ping"></div>

            {/* Central droplet */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="relative">
                <Droplet className="text-blue-600" size={32} />
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white rounded-full w-4 h-4"></div>
              </div>
            </div>

            {/* Orbiting dot */}
            <div className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-3 h-3 bg-blue-500 rounded-full">
              <div className="animate-orbit"></div>
            </div>
          </div>

          <div className="animate-pulse">
            <h1 className="text-2xl font-bold text-gray-800 mb-2">TruthLens</h1>
            <p className="text-gray-600">Loading your dashboard...</p>
          </div>
        </div>
      </div>
    );
  };


  // Add this function to analyze prompts as the user types
  const analyzePrompt = async (promptText) => {
    // Skip processing for short inputs
    if (!promptText || promptText.trim().length < 15) {
      setShowPromptSuggestions(false);
      return;
    }

    // Skip API calls during typing for URLs (which don't need prompt suggestions)
    if (promptText.trim().toLowerCase().startsWith("http")) {
      setShowPromptSuggestions(false);
      return;
    }

    try {
      // Use AbortController to cancel pending requests when a new one starts
      const controller = new AbortController();
      const signal = controller.signal;

      // Store the controller reference to cancel it if needed
      window.lastPromptController = controller;

      const response = await fetch(`${apiBaseUrl}/analyze-prompt`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: promptText,
        }),
        signal: signal,
      });

      if (!response.ok) {
        throw new Error(`API responded with status: ${response.status}`);
      }

      const data = await response.json();

      if (
        !data.is_good_prompt &&
        data.suggestions &&
        data.suggestions.length > 0
      ) {
        setPromptSuggestions(data.suggestions);
        setImprovedPrompt(data.improved_prompt);
        setShowPromptSuggestions(true);
      } else {
        setShowPromptSuggestions(false);
      }
    } catch (err) {
      // Ignore aborted fetch errors (these are expected when typing quickly)
      if (err.name !== "AbortError") {
        console.error("Error analyzing prompt:", err);
      }
      setShowPromptSuggestions(false);
    }
  };

  // Handle claim input with debounce
  const handleClaimChange = (e) => {
    const newClaim = e.target.value;
    setClaim(newClaim);

    // Clear previous timeout with a more significant delay
    if (promptAnalysisTimeout) {
      clearTimeout(promptAnalysisTimeout);
    }

    // Increase debounce time to reduce frequency of analyses
    const newTimeout = setTimeout(() => {
      // Only analyze if text is substantial to avoid unnecessary processing
      if (newClaim.trim().length > 15) {
        analyzePrompt(newClaim);
      } else {
        setShowPromptSuggestions(false);
      }
    }, 1200); // Increased from 800ms to 1200ms

    setPromptAnalysisTimeout(newTimeout);
  };

  // Function to use the improved prompt
  const useImprovedPrompt = () => {
    if (improvedPrompt) {
      setClaim(improvedPrompt);
      setShowPromptSuggestions(false);
    }
  };

  const TrustBadge3D = ({ score }) => {
    const canvasRef = React.useRef(null);

    React.useEffect(() => {
      // Capture the ref value at the time the effect runs
      const currentCanvas = canvasRef.current;
      if (!currentCanvas) return;

      // Set up Three.js scene
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);

      const renderer = new THREE.WebGLRenderer({
        canvas: currentCanvas,
        antialias: true,
        alpha: true,
      });

      renderer.setClearColor(0x000000, 0);
      renderer.setSize(150, 150);

      // Create badge shape
      const badgeGeometry = new THREE.CylinderGeometry(1, 1, 0.2, 32);

      // Color based on score
      let badgeColor;
      if (score >= 0.8) badgeColor = 0x10b981; // Green
      else if (score >= 0.5) badgeColor = 0xf59e0b; // Yellow
      else badgeColor = 0xef4444; // Red

      const badgeMaterial = new THREE.MeshStandardMaterial({
        color: badgeColor,
        metalness: 0.7,
        roughness: 0.3,
      });

      const badge = new THREE.Mesh(badgeGeometry, badgeMaterial);
      scene.add(badge);

      // Add checkmark or X embossing based on score
      const embossGeometry =
        score >= 0.5
          ? new THREE.TorusGeometry(0.5, 0.1, 16, 100, Math.PI * 0.5)
          : new THREE.CylinderGeometry(0.05, 0.05, 1, 32);

      const embossMaterial = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        metalness: 0.5,
        roughness: 0.5,
      });

      const emboss = new THREE.Mesh(embossGeometry, embossMaterial);

      if (score >= 0.5) {
        emboss.position.y = 0.11;
        emboss.rotation.z = Math.PI * 0.25;
      } else {
        emboss.position.y = 0.11;
        emboss.rotation.z = Math.PI * 0.25;
      }

      badge.add(emboss);

      // Add lights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
      scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
      directionalLight.position.set(5, 5, 5);
      scene.add(directionalLight);

      camera.position.z = 3;

      // Animation loop
      let animationFrameId;
      const animate = () => {
        animationFrameId = requestAnimationFrame(animate);

        badge.rotation.y += 0.01;

        renderer.render(scene, camera);
      };

      animate();

      // Handle cleanup
      return () => {
        cancelAnimationFrame(animationFrameId);

        // Dispose THREE.js resources
        badgeGeometry.dispose();
        badgeMaterial.dispose();
        embossGeometry.dispose();
        embossMaterial.dispose();

        // We don't need to remove the canvas since we're using the existing canvas element
        renderer.dispose();
      };
    }, [score]);

    return (
      <div className="flex flex-col items-center">
        <canvas ref={canvasRef} className="w-[150px] h-[150px]" />
        <div className="mt-2 font-bold text-lg">
          {(score * 100).toFixed(0)}% Trust Score
        </div>
      </div>
    );
  };

  const handleAnalyze = async () => {
    try {
      setIsAnalyzing(true);
      setError(null);

      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          claim: claim,
          use_rag: useRag,
          use_kg: useKg,
          is_url: claim.trim().toLowerCase().startsWith("http"),
        }),
      });

      const data = await response.json();
      console.log("Response from backend:", data);

      if (data.error) {
        setError(data.error);
        return;
      }

      // Save to history if analysis was successful
      if (data.status === "success") {
        saveToHistory(data);
      }

      setResult(data);
      // Show the animation when the result is ready
      setShowResultsAnimation(true);
    } catch (error) {
      console.error("Error during analysis:", error);
      setError("Terjadi kesalahan saat menganalisis klaim. Silakan coba lagi.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const ParticleBackground = () => {
    const canvasRef = React.useRef(null);
    const animationRef = React.useRef(null);
    const particlesRef = React.useRef([]);
    const initialized = React.useRef(false);

    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d", { alpha: true });
      let width = window.innerWidth;
      let height = window.innerHeight;

      // Only initialize particles once
      if (!initialized.current) {
        canvas.width = width;
        canvas.height = height;

        // Further reduce particle count for better performance
        const particleCount = 15; // Reduced from 25

        for (let i = 0; i < particleCount; i++) {
          particlesRef.current.push({
            x: Math.random() * width,
            y: Math.random() * height,
            size: Math.random() * 2 + 1, // Even smaller particles
            speedX: Math.random() * 0.3 - 0.15, // Even slower movement
            speedY: Math.random() * 0.3 - 0.15,
            color: `rgba(74, 144, 226, ${Math.random() * 0.2 + 0.1})`, // Reduced opacity
          });
        }

        initialized.current = true;
      }

      // Further reduce connection distance
      const connectionDistance = 100; // Reduced from 120

      // Even lower frame rate
      let lastTime = 0;
      const targetFPS = 15; // Reduced from 20
      const interval = 1000 / targetFPS;

      // Pre-calculate some values to avoid calculations in render loop
      const particles = particlesRef.current;
      const particleCount = particles.length;

      // Use a visibility check to avoid rendering offscreen particles
      const isVisible = () => document.visibilityState === "visible";

      const animate = (currentTime) => {
        // Skip animation when tab is not visible
        if (!isVisible()) {
          animationRef.current = requestAnimationFrame(animate);
          return;
        }

        const deltaTime = currentTime - lastTime;
        if (deltaTime < interval) {
          animationRef.current = requestAnimationFrame(animate);
          return;
        }

        lastTime = currentTime - (deltaTime % interval);

        ctx.clearRect(0, 0, width, height);

        // Update and draw particles
        for (let i = 0; i < particleCount; i++) {
          const particle = particles[i];

          particle.x += particle.speedX;
          particle.y += particle.speedY;

          // Simplified wrap-around
          if (particle.x > width) particle.x = 0;
          else if (particle.x < 0) particle.x = width;
          if (particle.y > height) particle.y = 0;
          else if (particle.y < 0) particle.y = height;

          // Only draw particles that are in the viewport
          if (
            particle.x >= 0 &&
            particle.x <= width &&
            particle.y >= 0 &&
            particle.y <= height
          ) {
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
            ctx.fillStyle = particle.color;
            ctx.fill();
          }
        }

        // Only connect every third particle to every fourth particle for much fewer connections
        for (let i = 0; i < particleCount; i += 3) {
          const particleA = particles[i];
          for (let j = i + 4; j < particleCount; j += 4) {
            const particleB = particles[j];

            // Quick distance check using manhattan distance first (faster)
            const dx = Math.abs(particleA.x - particleB.x);
            const dy = Math.abs(particleA.y - particleB.y);

            if (dx + dy < connectionDistance * 1.5) {
              // Now do the more expensive Euclidean distance calculation
              const distance = Math.sqrt(dx * dx + dy * dy);

              if (distance < connectionDistance) {
                ctx.beginPath();
                ctx.moveTo(particleA.x, particleA.y);
                ctx.lineTo(particleB.x, particleB.y);
                // Simplified opacity calculation
                ctx.strokeStyle = `rgba(74, 144, 226, ${
                  0.1 * (1 - distance / connectionDistance)
                })`;
                ctx.lineWidth = 1;
                ctx.stroke();
              }
            }
          }
        }

        animationRef.current = requestAnimationFrame(animate);
      };

      animationRef.current = requestAnimationFrame(animate);

      // Heavily debounced resize handler
      let resizeTimeout;
      const handleResize = () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
          width = window.innerWidth;
          height = window.innerHeight;
          canvas.width = width;
          canvas.height = height;
        }, 500); // Increased from 200ms to 500ms
      };

      window.addEventListener("resize", handleResize, { passive: true });

      // Add visibility change listener to pause animation when tab is not visible
      document.addEventListener("visibilitychange", () => {
        if (isVisible() && !animationRef.current) {
          animationRef.current = requestAnimationFrame(animate);
        }
      });

      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
        window.removeEventListener("resize", handleResize);
        document.removeEventListener("visibilitychange", () => {});
        clearTimeout(resizeTimeout);
      };
    }, []);

    return (
      <canvas
        ref={canvasRef}
        className="fixed top-0 left-0 w-full h-full -z-10 opacity-30"
      />
    );
  };

  // Optimized CredibilityHeatmap Component
  const CredibilityHeatmap = ({ score, level }) => {
    const [showDetails, setShowDetails] = React.useState(false);
    const [animatedScore, setAnimatedScore] = React.useState(0);

    // Get credibility level based on score
    const getCredibilityLevelFromScore = React.useCallback((scoreValue) => {
      if (scoreValue >= 0.8) return "HIGH";
      if (scoreValue >= 0.6) return "MEDIUM_HIGH";
      if (scoreValue >= 0.4) return "MEDIUM";
      if (scoreValue >= 0.2) return "LOW_MEDIUM";
      return "LOW";
    }, []);

    // Get dynamic level based on either provided level or calculated from score
    const dynamicLevel = React.useMemo(() => {
      return level || getCredibilityLevelFromScore(score || 0);
    }, [level, score, getCredibilityLevelFromScore]);

    // Get credibility details based on level
    const getCredibilityDetails = React.useCallback((currentLevel) => {
      const details = {
        HIGH: {
          title: "High Credibility",
          description:
            "This source demonstrates strong journalistic standards with accurate information from reputable outlets.",
          characteristics: [
            "Transparent author credentials",
            "Cites primary sources",
            "Minimal emotional language",
            "History of accurate reporting",
          ],
          color: "#10b981",
          lightColor: "#d1fae5",
        },
        MEDIUM_HIGH: {
          title: "Medium-High Credibility",
          description:
            "Generally reliable source with occasional minor issues in reporting.",
          characteristics: [
            "Identified authors",
            "Mostly neutral language",
            "Some citation of sources",
            "Largely accurate track record",
          ],
          color: "#22c55e",
          lightColor: "#dcfce7",
        },
        MEDIUM: {
          title: "Medium Credibility",
          description:
            "Mixed reliability with both accurate and potentially misleading content.",
          characteristics: [
            "Some emotional language",
            "Inconsistent source attribution",
            "Occasional factual errors",
            "Some partisan framing",
          ],
          color: "#f59e0b",
          lightColor: "#fef3c7",
        },
        LOW_MEDIUM: {
          title: "Low-Medium Credibility",
          description:
            "Concerning reliability issues with frequent problems in reporting.",
          characteristics: [
            "Heavy use of emotional language",
            "Limited source attribution",
            "History of misleading claims",
            "Strong partisan bias",
          ],
          color: "#f97316",
          lightColor: "#ffedd5",
        },
        LOW: {
          title: "Low Credibility",
          description:
            "Serious reliability concerns with evidence of misleading or false information.",
          characteristics: [
            "Anonymous authors",
            "Highly emotional language",
            "No source citations",
            "History of false claims",
          ],
          color: "#ef4444",
          lightColor: "#fee2e2",
        },
      };

      return (
        details[currentLevel] || {
          title: "Unknown Credibility",
          description: "Insufficient data to determine credibility level.",
          characteristics: [
            "Unknown source",
            "Unable to verify credentials",
            "Limited publication history",
          ],
          color: "#9ca3af",
          lightColor: "#f3f4f6",
        }
      );
    }, []);

    // Get credibility details for the current level
    const credibilityDetails = React.useMemo(
      () => getCredibilityDetails(dynamicLevel),
      [getCredibilityDetails, dynamicLevel]
    );

    // Helper function to convert hex to RGBA
    const hexToRgba = React.useCallback((hex, alpha = 1) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result
        ? `rgba(${parseInt(result[1], 16)}, ${parseInt(
            result[2],
            16
          )}, ${parseInt(result[3], 16)}, ${alpha})`
        : `rgba(0, 0, 0, ${alpha})`;
    }, []);

    // Generate the heatmap cells data based on score
    const generateHeatmapData = React.useCallback((credScore) => {
      const numCells = 20; // 20 cells for 5% increments
      const cells = [];

      for (let i = 0; i < numCells; i++) {
        const cellScore = (i + 1) / numCells;
        let color;

        if (cellScore <= 0.2) {
          color = "#ef4444"; // Low - Red
        } else if (cellScore <= 0.4) {
          color = "#f97316"; // Low-Medium - Orange
        } else if (cellScore <= 0.6) {
          color = "#f59e0b"; // Medium - Amber
        } else if (cellScore <= 0.8) {
          color = "#22c55e"; // Medium-High - Light Green
        } else {
          color = "#10b981"; // High - Green
        }

        cells.push({
          index: i,
          value: cellScore,
          active: cellScore <= credScore,
          color: color,
          alpha: cellScore <= credScore ? 1 : 0.15,
        });
      }

      return cells;
    }, []);

    // Animate the score value on component mount or when score changes
    React.useEffect(() => {
      const targetScore = score || 0;
      let startTimestamp;
      const duration = 1500; // 1.5 seconds animation

      const animateScore = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const elapsed = timestamp - startTimestamp;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function for smoother animation
        const eased =
          progress < 0.5
            ? 4 * progress * progress * progress
            : 1 - Math.pow(-2 * progress + 2, 3) / 2;

        setAnimatedScore(targetScore * eased);

        if (progress < 1) {
          requestAnimationFrame(animateScore);
        }
      };

      const animationId = requestAnimationFrame(animateScore);

      return () => {
        cancelAnimationFrame(animationId);
      };
    }, [score]);

    // Generate heatmap cells based on the animated score
    const heatmapCells = React.useMemo(
      () => generateHeatmapData(animatedScore),
      [generateHeatmapData, animatedScore]
    );

    return (
      <div className="relative">
        <h3 className="text-sm text-gray-500 mb-2">Source Credibility</h3>

        {/* Heatmap Visualization */}
        <div
          className="mb-2 relative"
          onClick={() => setShowDetails(!showDetails)}
          title="Click for more details"
        >
          {/* Credibility Scale Labels */}
          <div className="flex justify-between mb-1 text-xs">
            <span className="text-red-500 font-medium">Low</span>
            <span className="text-yellow-500 font-medium">Medium</span>
            <span className="text-green-500 font-medium">High</span>
          </div>

          {/* Heatmap Cells */}
          <div className="flex h-10 rounded-lg overflow-hidden transition-all duration-300 hover:shadow-md cursor-pointer">
            {heatmapCells.map((cell) => (
              <div
                key={cell.index}
                className="flex-1 transition-all duration-300 flex items-center justify-center relative"
                style={{
                  backgroundColor: hexToRgba(cell.color, cell.alpha),
                  height: cell.active ? "100%" : "80%",
                  alignSelf: cell.active ? "flex-start" : "center",
                }}
              >
                {cell.value === 0.25 && (
                  <div className="absolute top-full text-xs text-gray-500 mt-1">
                    25%
                  </div>
                )}
                {cell.value === 0.5 && (
                  <div className="absolute top-full text-xs text-gray-500 mt-1">
                    50%
                  </div>
                )}
                {cell.value === 0.75 && (
                  <div className="absolute top-full text-xs text-gray-500 mt-1">
                    75%
                  </div>
                )}
              </div>
            ))}

            {/* Current Value Indicator */}
            <div
              className="absolute top-0 h-full transition-all duration-300"
              style={{
                left: `${Math.min(Math.max(animatedScore * 100, 0), 100)}%`,
                transform: "translateX(-50%)",
              }}
            >
              <div className="w-px h-full bg-gray-800"></div>
              <div className="w-3 h-3 bg-white border-2 border-gray-800 rounded-full -mt-1.5 transform translate-x(-50%)"></div>
            </div>
          </div>

          {/* Score & Level Display */}
          <div className="mt-5 flex justify-between items-center">
            <div
              className="text-3xl font-bold"
              style={{ color: credibilityDetails.color }}
            >
              {Math.round(animatedScore * 100)}%
            </div>
            <div
              className="px-3 py-1 rounded-full text-sm font-medium"
              style={{
                backgroundColor: hexToRgba(credibilityDetails.color, 0.15),
                color: credibilityDetails.color,
              }}
            >
              {credibilityDetails.title}
            </div>
          </div>

          <div className="mt-2 text-sm text-gray-600">
            {credibilityDetails.description.split(".")[0]}.
          </div>

          <div className="text-xs text-center text-gray-500 mt-2 animate-pulse">
            Click for more details
          </div>
        </div>

        {/* Detailed Information Panel */}
        {showDetails && (
          <div className="mt-4 bg-gray-50 border border-gray-200 rounded-lg p-4 animate-fade-in transition-all duration-300">
            <div className="flex justify-between items-center mb-2">
              <h4
                className="font-medium text-lg"
                style={{ color: credibilityDetails.color }}
              >
                {credibilityDetails.title}
              </h4>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  setShowDetails(false);
                }}
                className="text-gray-500 hover:text-gray-700"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </div>

            <p className="text-gray-600 mb-3">
              {credibilityDetails.description}
            </p>

            <div className="mt-3">
              <h5 className="font-medium text-gray-700 mb-2">
                Characteristics:
              </h5>
              <ul className="space-y-1">
                {credibilityDetails.characteristics.map((item, index) => (
                  <li key={index} className="flex items-start">
                    <span
                      className="inline-block w-2 h-2 mt-1.5 mr-2 rounded-full"
                      style={{ backgroundColor: credibilityDetails.color }}
                    ></span>
                    <span className="text-gray-600">{item}</span>
                  </li>
                ))}
              </ul>
            </div>

            {/* Credibility Factors */}
            <div
              className="mt-4 p-3 rounded-lg text-sm"
              style={{
                backgroundColor: hexToRgba(credibilityDetails.color, 0.1),
                borderLeft: `3px solid ${credibilityDetails.color}`,
              }}
            >
              <span
                className="font-medium"
                style={{ color: credibilityDetails.color }}
              >
                Recommendation:
              </span>{" "}
              {dynamicLevel === "HIGH" || dynamicLevel === "MEDIUM_HIGH" ? (
                <span className="text-gray-700">
                  This source appears reliable for fact-based information.
                </span>
              ) : dynamicLevel === "MEDIUM" ? (
                <span className="text-gray-700">
                  Verify key claims from this source with additional evidence.
                </span>
              ) : (
                <span className="text-gray-700">
                  Treat information from this source with significant caution.
                </span>
              )}
            </div>

            {/* Credibility Score Distribution */}
            <div className="mt-4">
              <h5 className="font-medium text-gray-700 mb-2">
                Credibility Analysis:
              </h5>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-gray-600">
                      Source Reputation
                    </span>
                    <span className="text-sm text-gray-700">
                      {Math.round(
                        (animatedScore * 0.9 + Math.random() * 0.1) * 100
                      )}
                      %
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-2 rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.round(
                          (animatedScore * 0.9 + Math.random() * 0.1) * 100
                        )}%`,
                        backgroundColor: credibilityDetails.color,
                      }}
                    ></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-gray-600">
                      Factual Accuracy
                    </span>
                    <span className="text-sm text-gray-700">
                      {Math.round(
                        (animatedScore * 0.95 + Math.random() * 0.05) * 100
                      )}
                      %
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-2 rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.round(
                          (animatedScore * 0.95 + Math.random() * 0.05) * 100
                        )}%`,
                        backgroundColor: credibilityDetails.color,
                      }}
                    ></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-gray-600">
                      Source Transparency
                    </span>
                    <span className="text-sm text-gray-700">
                      {Math.round(
                        (animatedScore * 0.85 + Math.random() * 0.15) * 100
                      )}
                      %
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-2 rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.round(
                          (animatedScore * 0.85 + Math.random() * 0.15) * 100
                        )}%`,
                        backgroundColor: credibilityDetails.color,
                      }}
                    ></div>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm text-gray-600">Neutrality</span>
                    <span className="text-sm text-gray-700">
                      {Math.round(
                        (animatedScore * 0.8 + Math.random() * 0.2) * 100
                      )}
                      %
                    </span>
                  </div>
                  <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div
                      className="h-2 rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.round(
                          (animatedScore * 0.8 + Math.random() * 0.2) * 100
                        )}%`,
                        backgroundColor: credibilityDetails.color,
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const saveToHistory = (result) => {
    // Create a new history item with timestamp
    const historyItem = {
      id: Date.now(),
      timestamp: new Date().toLocaleString(),
      claim: result.claim,
      verdict: result.verdict,
      confidence: result.confidence,
      isUrl: result.is_url_input || false,
    };

    // Add to history
    setAnalysisHistory((prev) => [historyItem, ...prev].slice(0, 10)); // Keep only last 10 items
  };

  // Add this component inside your TruthLensApp component but outside the return statement
  const HistoryPanel = ({ isVisible, onClose, history, onSelectItem }) => {
    if (!isVisible) return null;

    return (
      <div className="fixed inset-y-0 right-0 max-w-sm w-full bg-white shadow-lg z-30 animate-slide-in-right">
        <div className="p-4 border-b border-gray-200 flex justify-between items-center">
          <h3 className="text-lg font-semibold">Analysis History</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        <div className="p-4 overflow-y-auto max-h-screen">
          {history.length === 0 ? (
            <div className="text-center text-gray-500 py-8">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-12 w-12 mx-auto text-gray-300 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <p>No analysis history yet</p>
            </div>
          ) : (
            <div className="space-y-3">
              {history.map((item) => (
                <div
                  key={item.id}
                  onClick={() => onSelectItem(item)}
                  className="p-3 border border-gray-200 rounded-lg hover:bg-blue-50 cursor-pointer transition"
                >
                  <div className="flex justify-between mb-1">
                    <div className="text-xs text-gray-500">
                      {item.timestamp}
                    </div>
                    {renderVerdictBadge(item.verdict)}
                  </div>
                  <div className="text-sm font-medium mb-1 truncate">
                    {item.claim}
                  </div>
                  <div className="flex items-center text-xs text-gray-500">
                    <div className="flex items-center mr-3">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-3 w-3 mr-1"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                      {item.confidence}% confidence
                    </div>
                    {item.isUrl && (
                      <div className="flex items-center">
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-3 w-3 mr-1"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
                          />
                        </svg>
                        URL Analysis
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };

  // Add this function after your other functions
  const handleHistoryItemSelect = (item) => {
    // Set the claim from history
    setClaim(item.claim);

    // Close the history panel
    setShowHistory(false);

    // Optionally, you could re-analyze the claim
    // analyzeClaim();
  };

  // Add after your other functions
  const generatePDF = () => {
    if (!result) return;

    // Import jsPDF (this is a dynamic import so it won't load until used)
    import("jspdf").then(({ default: jsPDF }) => {
      const doc = new jsPDF();

      // Add title
      doc.setFontSize(20);
      doc.text("TruthLens Analysis Report", 20, 20);

      // Add date
      doc.setFontSize(10);
      doc.setTextColor(100);
      doc.text(`Generated on: ${new Date().toLocaleString()}`, 20, 30);

      // Add claim
      doc.setFontSize(12);
      doc.setTextColor(0);
      doc.text("Claim:", 20, 45);

      // Wrap text for long claims
      const textLines = doc.splitTextToSize(result.claim, 170);
      doc.text(textLines, 20, 52);

      let yPos = 52 + textLines.length * 7;

      // Add verdict
      doc.setFontSize(14);

      if (result.verdict === "FAKE" || result.verdict === "False") {
        doc.setTextColor(255, 0, 0);
      } else if (result.verdict === "REAL" || result.verdict === "True") {
        doc.setTextColor(0, 128, 0);
      } else {
        doc.setTextColor(255, 165, 0);
      }

      doc.text(`Verdict: ${result.verdict}`, 20, yPos);
      yPos += 10;

      // Add confidence
      doc.setFontSize(12);
      doc.setTextColor(0);
      doc.text(`Confidence: ${result.confidence}%`, 20, yPos);
      yPos += 15;

      // Add explanation
      doc.text("Analysis:", 20, yPos);
      yPos += 7;

      const explanationLines = doc.splitTextToSize(result.explanation, 170);
      doc.text(explanationLines, 20, yPos);
      yPos += explanationLines.length * 7 + 10;

      // Add emotional manipulation
      doc.text("Emotional Manipulation:", 20, yPos);
      yPos += 7;

      const emotionalLines = doc.splitTextToSize(
        result.emotional_manipulation.explanation,
        170
      );
      doc.text(emotionalLines, 20, yPos);
      yPos += emotionalLines.length * 7 + 10;

      // Add entities
      if (result.entities && Object.keys(result.entities).length > 0) {
        doc.text("Detected Entities:", 20, yPos);
        yPos += 7;

        Object.entries(result.entities).forEach(([entityType, entityList]) => {
          doc.text(`${entityType}: ${entityList.join(", ")}`, 30, yPos);
          yPos += 7;
        });
        yPos += 3;
      }

      // Add source information if available
      if (result.source_metadata) {
        doc.text("Source Information:", 20, yPos);
        yPos += 7;

        doc.text(`Title: ${result.source_metadata.title}`, 30, yPos);
        yPos += 7;
        doc.text(`Author: ${result.source_metadata.author}`, 30, yPos);
        yPos += 7;
        doc.text(
          `Published: ${result.source_metadata.published_date}`,
          30,
          yPos
        );
        yPos += 7;
        doc.text(`Domain: ${result.source_metadata.domain}`, 30, yPos);
        yPos += 10;
      }

      // Add footer
      doc.setFontSize(10);
      doc.setTextColor(100);
      doc.text(
        "Generated by TruthLens - Advanced Fake News Detection System",
        20,
        280
      );

      // Save the PDF
      doc.save("TruthLens-Analysis.pdf");
    });
  };

  // Add a new component to display contradiction analysis
  const TitleContentContradictionWidget = ({ contradiction }) => {
    if (!contradiction) return null;

    const getSeverityColor = (severity) => {
      if (severity < 0.3) return "text-green-600";
      if (severity < 0.7) return "text-yellow-600";
      return "text-red-600";
    };

    return (
      <div className="bg-gray-50 p-3 rounded-lg">
        <h4 className="font-medium mb-2">Title-Content Analysis</h4>
        <div className={`text-sm ${getSeverityColor(contradiction.severity)}`}>
          <div>
            <strong>Contradiction Type:</strong> {contradiction.type}
          </div>
          <div>
            <strong>Severity:</strong>{" "}
            {(contradiction.severity * 100).toFixed(0)}%
          </div>
          {contradiction.is_misleading && (
            <div className="mt-2 text-red-600 font-semibold">
              Potentially Misleading Content
            </div>
          )}
          <div className="mt-2 text-gray-700">
            <strong>Explanation:</strong> {contradiction.explanation}
          </div>
        </div>
      </div>
    );
  };

  const loadSample = (sample) => {
    setClaim(sample.claim);
  };

  const InteractiveAIDetectionBadge = ({ aiDetection }) => {
    const [isExpanded, setIsExpanded] = React.useState(false);

    if (!aiDetection) return null;

    // Extract the score first for use in fallback values
    const score =
      aiDetection.ai_score !== undefined
        ? aiDetection.ai_score
        : aiDetection.ai_likelihood !== undefined
        ? aiDetection.ai_likelihood
        : aiDetection.ai_detection && aiDetection.ai_detection.ai_score
        ? aiDetection.ai_detection.ai_score
        : 0;

    // Determine fallback emoji based on score
    let fallbackEmoji = "❓";
    if (score >= 0.6) fallbackEmoji = "🤖";
    if (score <= 0.4) fallbackEmoji = "👤";

    // Now normalize the data structure regardless of how it comes in
    const normalizedData = {
      score: score,
      verdict:
        aiDetection.ai_verdict ||
        (aiDetection.ai_detection && aiDetection.ai_detection.ai_verdict) ||
        "Unknown",
      reasoning:
        aiDetection.reasoning ||
        (aiDetection.ai_detection && aiDetection.ai_detection.reasoning) ||
        "No analysis available",
      category:
        aiDetection.content_category ||
        (aiDetection.ai_detection &&
          aiDetection.ai_detection.content_category) ||
        "Unknown",
      subcategory:
        aiDetection.content_subcategory ||
        (aiDetection.ai_detection &&
          aiDetection.ai_detection.content_subcategory) ||
        null,
      patterns:
        aiDetection.pattern_analysis ||
        (aiDetection.ai_detection &&
          aiDetection.ai_detection.pattern_analysis) ||
        null,
      emoji:
        aiDetection.emoji ||
        (aiDetection.ai_detection && aiDetection.ai_detection.emoji) ||
        fallbackEmoji,
      displayMessage:
        aiDetection.display_message ||
        (aiDetection.ai_detection &&
          aiDetection.ai_detection.display_message) ||
        "Unable to determine if content was written by AI or human",
      isAi:
        aiDetection.is_ai !== undefined
          ? aiDetection.is_ai
          : aiDetection.ai_detection &&
            aiDetection.ai_detection.is_ai !== undefined
          ? aiDetection.ai_detection.is_ai
          : null,
      confidencePercentage:
        aiDetection.confidence_percentage ||
        (aiDetection.ai_detection &&
          aiDetection.ai_detection.confidence_percentage) ||
        Math.round(score * 100),
      linguisticTraits:
        aiDetection.linguistic_traits ||
        (aiDetection.ai_detection &&
          aiDetection.ai_detection.linguistic_traits) ||
        null,
    };

    // Determine background color based on AI score
    const getBackgroundColor = () => {
      if (normalizedData.isAi === true) return "bg-red-50 border-red-100";
      if (normalizedData.isAi === false) return "bg-green-50 border-green-100";
      return "bg-gray-50 border-gray-100";
    };

    // Determine text color based on AI score
    const getTextColor = () => {
      if (normalizedData.isAi === true) return "text-red-600";
      if (normalizedData.isAi === false) return "text-green-600";
      return "text-gray-600";
    };

    const scorePercentage = normalizedData.confidencePercentage;

    return (
      <div
        className={`mb-6 border rounded-lg bg-white shadow-sm overflow-hidden transition-all duration-500 ${
          isExpanded ? "p-6" : "p-4"
        }`}
      >
        <div className="flex justify-between items-center mb-3">
          <h3 className="text-lg font-medium">AI Content Detection</h3>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            {isExpanded ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M14.707 12.707a1 1 0 01-1.414 0L10 9.414l-3.293 3.293a1 1 0 01-1.414-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 010 1.414z"
                  clipRule="evenodd"
                />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                  clipRule="evenodd"
                />
              </svg>
            )}
          </button>
        </div>

        {/* Large emoji and verdict headline */}
        <div
          className={`p-4 mb-4 rounded-lg flex items-center ${getBackgroundColor()} transition-transform duration-500 ${
            isExpanded ? "scale-100" : "hover:scale-105"
          }`}
          onClick={() => !isExpanded && setIsExpanded(true)}
        >
          <div className={`text-5xl mr-4 ${isExpanded ? "" : "animate-pulse"}`}>
            {normalizedData.emoji}
          </div>
          <div>
            <h4 className="font-bold text-lg">
              {normalizedData.displayMessage}
            </h4>
            <p className={`mt-1 ${getTextColor()}`}>{normalizedData.verdict}</p>
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex items-center">
            <span className="font-medium">AI Probability:</span>
            <div className="ml-2 flex items-center flex-1">
              <div className="w-full h-3 bg-gray-200 rounded-full mr-2 overflow-hidden">
                <div
                  className={`h-3 rounded-full ${
                    normalizedData.isAi === true
                      ? "bg-red-500"
                      : normalizedData.isAi === false
                      ? "bg-green-500"
                      : "bg-gray-500"
                  } transition-all duration-1000 ease-out`}
                  style={{ width: `${scorePercentage}%` }}
                ></div>
              </div>
              <span className={`font-bold ${getTextColor()}`}>
                {scorePercentage}%
              </span>
            </div>
          </div>

          {isExpanded && (
            <>
              {normalizedData.reasoning && (
                <div className="animate-fade-in">
                  <span className="font-medium">Analysis:</span>
                  <p className="mt-1 text-gray-700">
                    {normalizedData.reasoning}
                  </p>
                </div>
              )}

              {normalizedData.category && (
                <div className="animate-fade-in">
                  <span className="font-medium">Content Type:</span>
                  <span className="ml-2">{normalizedData.category}</span>
                  {normalizedData.subcategory && (
                    <span className="ml-1 text-gray-500">
                      ({normalizedData.subcategory})
                    </span>
                  )}
                </div>
              )}

              {/* Show linguistic traits if available */}
              {normalizedData.linguisticTraits && (
                <div className="mt-3 animate-fade-in">
                  <span className="font-medium mb-1 block">
                    Linguistic Analysis:
                  </span>
                  <div className="grid grid-cols-2 gap-2 mt-2 text-sm">
                    {normalizedData.linguisticTraits
                      .personal_pronoun_density !== undefined && (
                      <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                        <span>Personal Pronouns:</span>
                        <span
                          className={
                            normalizedData.linguisticTraits
                              .personal_pronoun_density > 0.02
                              ? "text-green-600"
                              : "text-red-600"
                          }
                        >
                          {(
                            normalizedData.linguisticTraits
                              .personal_pronoun_density * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                    )}
                    {normalizedData.linguisticTraits.contraction_density !==
                      undefined && (
                      <div className="flex items-center justify-between p-2 bg-gray-50 rounded">
                        <span>Contractions:</span>
                        <span
                          className={
                            normalizedData.linguisticTraits
                              .contraction_density > 0.01
                              ? "text-green-600"
                              : "text-red-600"
                          }
                        >
                          {(
                            normalizedData.linguisticTraits
                              .contraction_density * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Show detected AI patterns */}
              {normalizedData.patterns &&
                normalizedData.patterns.patterns_found && (
                  <div className="mt-2 border-t pt-2 animate-fade-in">
                    <span className="font-medium block mb-1">
                      AI Patterns Detected:
                    </span>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-1 text-xs">
                      {Object.entries(normalizedData.patterns.patterns_found)
                        .filter(([pattern, count]) => count > 0)
                        .sort(([, countA], [, countB]) => countB - countA)
                        .slice(0, 4) // Show top 4 patterns
                        .map(([pattern, count]) => (
                          <div key={pattern} className="flex items-center">
                            <div className="w-2 h-2 rounded-full bg-red-400 mr-2"></div>
                            <span className="text-gray-600 truncate">
                              {pattern.replace(/\\b|\(|\)|\?/g, "")}:{" "}
                            </span>
                            <span className="ml-1 font-medium">{count}</span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
            </>
          )}
        </div>

        <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-500">
          {isExpanded ? (
            "AI detection is based on linguistic patterns, structural analysis, and content evaluation."
          ) : (
            <button
              className="text-blue-500 hover:text-blue-700 transition-colors"
              onClick={() => setIsExpanded(true)}
            >
              Click to see detailed analysis
            </button>
          )}
        </div>
      </div>
    );
  };

  const renderSourceMetadata = (metadata) => {
    if (!metadata) {
      return null;
    }

    return (
      <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <h3 className="text-md font-semibold mb-2 text-blue-800">
          Source Information
        </h3>
        <div className="grid grid-cols-1 gap-2">
          <div className="flex items-center">
            <FileText size={16} className="text-blue-700 mr-2" />
            <span className="text-sm font-medium mr-2">Title:</span>
            <span className="text-sm">{metadata.title}</span>
          </div>
          <div className="flex items-center">
            <User size={16} className="text-blue-700 mr-2" />
            <span className="text-sm font-medium mr-2">Author:</span>
            <span className="text-sm">{metadata.author}</span>
          </div>
          <div className="flex items-center">
            <Calendar size={16} className="text-blue-700 mr-2" />
            <span className="text-sm font-medium mr-2">Published:</span>
            <span className="text-sm">{metadata.published_date}</span>
          </div>
          <div className="flex items-center">
            <Globe size={16} className="text-blue-700 mr-2" />
            <span className="text-sm font-medium mr-2">Domain:</span>
            <span className="text-sm">{metadata.domain}</span>
          </div>
        </div>
      </div>
    );
  };

  const VerificationBadge = ({ verificationPerformed }) => {
    if (!verificationPerformed) return null;

    return (
      <div className="flex items-center bg-indigo-50 text-indigo-800 text-xs px-2 py-1 rounded mb-4">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-4 w-4 mr-1"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
          />
        </svg>
        Advanced verification analysis performed
      </div>
    );
  };

  // Add a component to visualize emotional manipulation
  const EmotionalAnalysisChart = ({ sentimentAnalysis }) => {
    if (
      !sentimentAnalysis ||
      !sentimentAnalysis.details ||
      !sentimentAnalysis.details.emotion_keywords
    ) {
      return null;
    }

    // Prepare emotion data for bar chart
    const emotionData = Object.entries(
      sentimentAnalysis.details.emotion_keywords
    )
      .map(([emotion, score]) => ({
        emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
        score: score * 100, // Convert to percentage
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5); // Get top 5 emotions

    return (
      <div className="mb-6 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium mb-3">Emotional Content Analysis</h4>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={emotionData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 70, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                domain={[0, Math.max(...emotionData.map((d) => d.score)) * 1.2]}
              />
              <YAxis dataKey="emotion" type="category" />
              <Tooltip
                formatter={(value) => [`${value.toFixed(1)}%`, "Intensity"]}
              />
              <Bar dataKey="score" fill="#8884d8">
                {emotionData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={
                      entry.score > 10
                        ? "#ef4444"
                        : entry.score > 5
                        ? "#f59e0b"
                        : "#10b981"
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  // Add this component before your return statement in TruthLensApp:
  const EnhancedVerdictAssessment = ({ result }) => {
    if (!result) return null;

    // Extract key properties
    const verdict = result.verdict || result.fact_check?.verdict;
    const confidence = result.confidence || result.fact_check?.confidence;
    const explanation = result.explanation || result.fact_check?.explanation;

    // Determine verdict colors and labels
    const getVerdictInfo = () => {
      if (!verdict) {
        return {
          color: "gray",
          label: "Unverified",
          bgColor: "bg-gray-500",
          textColor: "text-gray-800",
        };
      }

      const verdictLower = verdict.toLowerCase();
      if (verdictLower === "true" || verdictLower === "real") {
        return {
          color: "green",
          label: "True",
          bgColor: "bg-green-500",
          textColor: "text-green-800",
        };
      } else if (
        verdictLower === "partially true" ||
        verdictLower === "misleading"
      ) {
        return {
          color: "yellow",
          label: "Partially True",
          bgColor: "bg-yellow-500",
          textColor: "text-yellow-800",
        };
      } else if (verdictLower === "false" || verdictLower === "fake") {
        return {
          color: "red",
          label: "False",
          bgColor: "bg-red-500",
          textColor: "text-red-800",
        };
      } else {
        return {
          color: "gray",
          label: "Unverified",
          bgColor: "bg-gray-500",
          textColor: "text-gray-800",
        };
      }
    };

    const verdictInfo = getVerdictInfo();

    // Generate detailed explanation based on verdict
    const getDetailedExplanation = () => {
      if (verdictInfo.label === "True") {
        return "This claim is well-supported by reliable evidence. Multiple credible sources confirm the information.";
      } else if (verdictInfo.label === "Partially True") {
        return "This claim contains some accurate information but is misleading or omits important context.";
      } else if (verdictInfo.label === "False") {
        return "This claim is contradicted by reliable evidence. Credible sources refute the information.";
      } else {
        return "There is insufficient evidence to determine the accuracy of this claim.";
      }
    };

    return (
      <div className="mb-6 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
        <div className="flex justify-between items-start mb-4">
          <h3 className="text-lg font-medium">Verdict Assessment</h3>
          <div
            className={`px-3 py-1 rounded-full ${verdictInfo.bgColor} bg-opacity-20 ${verdictInfo.textColor} font-medium`}
          >
            {verdictInfo.label}
          </div>
        </div>

        <div className="mb-4">
          <div className="flex justify-between items-center mb-1">
            <span className="text-sm text-gray-600">Confidence Level</span>
            <span className="text-sm font-medium">{confidence || 0}%</span>
          </div>
          <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={`h-3 ${verdictInfo.bgColor}`}
              style={{ width: `${confidence || 0}%` }}
            ></div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div
            className={`p-3 rounded-lg ${verdictInfo.bgColor} bg-opacity-10 border border-${verdictInfo.color}-200`}
          >
            <h4 className={`font-medium ${verdictInfo.textColor} mb-1`}>
              Verdict
            </h4>
            <p className="text-sm text-gray-700">{verdictInfo.label}</p>
          </div>

          <div className="p-3 rounded-lg bg-blue-50 border border-blue-200 md:col-span-2">
            <h4 className="font-medium text-blue-800 mb-1">
              Expert Assessment
            </h4>
            <p className="text-sm text-gray-700">
              {explanation || getDetailedExplanation()}
            </p>
          </div>
        </div>

        {result.verification_note && (
          <div className="p-3 rounded-lg bg-purple-50 border border-purple-200 mb-4">
            <h4 className="font-medium text-purple-800 mb-1">
              Verification Note
            </h4>
            <p className="text-sm text-gray-700">{result.verification_note}</p>
          </div>
        )}

        {result.reasoning && (
          <div>
            <h4 className="font-medium mb-2">Analysis Reasoning</h4>
            <div className="text-sm text-gray-700 bg-gray-50 p-3 rounded-lg whitespace-pre-wrap">
              {result.reasoning}
            </div>
          </div>
        )}
      </div>
    );
  };

  // Add this component before your return statement in TruthLensApp:
  const ConfidenceFactorsWidget = ({ factors }) => {
    if (!factors) return null;

    // Format the factors for better display
    const formatFactor = (value, label) => {
      const percentage = (value * 100).toFixed(0);
      let color = "text-gray-600";

      if (value > 0.7) color = "text-green-600";
      else if (value > 0.4) color = "text-yellow-600";
      else color = "text-red-600";

      return (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium">{label}</span>
          <span className={`text-sm ${color}`}>{percentage}%</span>
          <div className="w-24 h-2 bg-gray-200 rounded-full ml-2">
            <div
              className={`h-2 rounded-full ${
                value > 0.7
                  ? "bg-green-500"
                  : value > 0.4
                  ? "bg-yellow-500"
                  : "bg-red-500"
              }`}
              style={{ width: `${percentage}%` }}
            ></div>
          </div>
        </div>
      );
    };

    return (
      <div className="mb-4 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium mb-3">Confidence Factors</h4>
        <div className="space-y-1">
          {factors.semantic_similarity !== undefined &&
            formatFactor(factors.semantic_similarity, "Evidence Similarity")}
          {factors.entity_overlap !== undefined &&
            formatFactor(factors.entity_overlap, "Entity Match")}
          {factors.evidence_strength !== undefined &&
            formatFactor(factors.evidence_strength, "Evidence Strength")}
          {factors.source_credibility !== undefined &&
            formatFactor(factors.source_credibility, "Source Credibility")}
        </div>
        <div className="mt-3 text-xs text-gray-500">
          These factors were used to calibrate the confidence score
        </div>
      </div>
    );
  };

  const renderCredibilityAssessment = (assessment) => {
    if (!assessment) {
      return null;
    }

    return (
      <div className="mt-4">
        <div className="mb-3">
          <CredibilityHeatmap
            score={assessment.score}
            level={assessment.level}
          />
          <br></br>
        </div>

        <div className="mb-3">
          <p className="text-sm">{assessment.explanation}</p>
        </div>

        <div className="text-sm text-gray-700">
          <h4 className="font-medium mb-2">Credibility Factors:</h4>
          <ul className="list-disc pl-5 space-y-1">
            {assessment.details.map((detail, index) => (
              <li key={index}>{detail}</li>
            ))}
          </ul>
        </div>
      </div>
    );
  };

  const renderVerdictBadge = (verdict) => {
    // Normalize verdict text to handle different formats
    const normalizedVerdict = verdict ? verdict.toLowerCase().trim() : "";

    // True verdicts
    if (normalizedVerdict === "real" || normalizedVerdict === "true") {
      return (
        <div className="relative">
          <div className="bg-green-100 text-green-800 rounded-full px-3 py-1 flex items-center">
            <Check size={16} className="mr-1" />
            Reliable
          </div>
          <div className="absolute inset-0 bg-green-100 rounded-full animate-pulse-slow opacity-60"></div>
        </div>
      );
    }
    // Partially true verdicts
    else if (
      normalizedVerdict === "misleading" ||
      normalizedVerdict === "partially true" ||
      normalizedVerdict.includes("partial")
    ) {
      return (
        <div className="relative">
          <div className="bg-yellow-100 text-yellow-800 rounded-full px-3 py-1 flex items-center">
            <AlertTriangle size={16} className="mr-1" />
            Misleading
          </div>
          <div className="absolute inset-0 bg-yellow-100 rounded-full animate-pulse-slow opacity-60"></div>
        </div>
      );
    }
    // False verdicts
    else if (normalizedVerdict === "fake" || normalizedVerdict === "false") {
      return (
        <div className="relative">
          <div className="bg-red-100 text-red-800 rounded-full px-3 py-1 flex items-center">
            <AlertCircle size={16} className="mr-1" />
            Fake News
          </div>
          <div className="absolute inset-0 bg-red-100 rounded-full animate-pulse-slow opacity-60"></div>
        </div>
      );
    }
    // Unverified/unknown verdicts
    else {
      return (
        <div className="bg-gray-100 text-gray-800 rounded-full px-3 py-1 flex items-center">
          <Info size={16} className="mr-1" />
          Unverified
        </div>
      );
    }
  };

  // Add this component inside your TruthLensApp component but outside the return statement
  const PromptSuggestionPanel = ({
    prompt,
    suggestions,
    onUseImproved,
    isVisible,
  }) => {
    if (!isVisible || !suggestions || suggestions.length === 0) return null;

    return (
      <div className="mt-2 p-3 bg-blue-50 rounded-lg border border-blue-100 animate-fade-in">
        <div className="flex justify-between items-start">
          <div className="flex items-center">
            <Info size={16} className="text-blue-600 mr-2 flex-shrink-0" />
            <h4 className="font-medium text-blue-800">Prompt Suggestions</h4>
          </div>
          {onUseImproved && (
            <button
              onClick={onUseImproved}
              className="text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700"
            >
              Use Improved Prompt
            </button>
          )}
        </div>
        <ul className="mt-2 space-y-1 text-sm text-blue-700">
          {suggestions.map((suggestion, index) => (
            <li key={index} className="flex items-start">
              <ChevronRight size={14} className="mr-1 mt-1 flex-shrink-0" />
              <span>{suggestion}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  };

  // Optimized WaterWaveEffect Component
  const WaterWaveEffect = () => {
    const canvasRef = React.useRef(null);
    const animationRef = React.useRef(null);
    const initialized = React.useRef(false);
    const wavesRef = React.useRef([]);

    React.useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d", { alpha: true });

      let width = canvas.offsetWidth;
      let height = canvas.offsetHeight;

      canvas.width = width;
      canvas.height = height;

      // Initialize waves only once
      if (!initialized.current) {
        // Only use 1 wave for better performance (reduced from 2)
        wavesRef.current = [
          {
            frequency: 0.04,
            amplitude: 12, // Reduced amplitude
            speed: 0.01, // Slower speed
            color: "rgba(255, 255, 255, 0.05)", // Reduced opacity
            offset: Math.random() * Math.PI * 2,
          },
        ];

        initialized.current = true;
      }

      // Even lower frame rate
      let lastTime = 0;
      const targetFPS = 12; // Further reduced from 24
      const interval = 1000 / targetFPS;

      const waves = wavesRef.current;

      // Pre-compute values for optimization
      const step = Math.ceil(width / 50); // Increased step size for fewer points

      // Visibility check to avoid rendering when tab is not visible
      const isVisible = () => document.visibilityState === "visible";

      const drawWave = (wave, time) => {
        ctx.beginPath();

        const { frequency, amplitude, speed, offset } = wave;

        for (let x = 0; x < width; x += step) {
          const y =
            height -
            Math.sin(x * frequency + time * speed + offset) * amplitude -
            20;

          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        // Add the last point
        const x = width - 1;
        const y =
          height -
          Math.sin(x * frequency + time * speed + offset) * amplitude -
          20;
        ctx.lineTo(x, y);

        // Close the path
        ctx.lineTo(width, height);
        ctx.lineTo(0, height);
        ctx.closePath();

        ctx.fillStyle = wave.color;
        ctx.fill();
      };

      let startTime = Date.now();

      const animate = (currentTime) => {
        // Skip animation when tab is not visible
        if (!isVisible()) {
          animationRef.current = requestAnimationFrame(animate);
          return;
        }

        const deltaTime = currentTime - lastTime;
        if (deltaTime < interval) {
          animationRef.current = requestAnimationFrame(animate);
          return;
        }

        lastTime = currentTime - (deltaTime % interval);

        ctx.clearRect(0, 0, width, height);

        const elapsed = Date.now() - startTime;

        waves.forEach((wave) => {
          drawWave(wave, elapsed);
        });

        animationRef.current = requestAnimationFrame(animate);
      };

      animationRef.current = requestAnimationFrame(animate);

      // Heavily debounced resize handler
      let resizeTimeout;
      const handleResize = () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
          width = canvas.offsetWidth;
          height = canvas.offsetHeight;
          canvas.width = width;
          canvas.height = height;
        }, 500); // Increased from 200ms to 500ms
      };

      window.addEventListener("resize", handleResize, { passive: true });

      // Add visibility change listener
      document.addEventListener("visibilitychange", () => {
        if (isVisible() && !animationRef.current) {
          animationRef.current = requestAnimationFrame(animate);
        }
      });

      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
        }
        window.removeEventListener("resize", handleResize);
        document.removeEventListener("visibilitychange", () => {});
        clearTimeout(resizeTimeout);
      };
    }, []);

    return (
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full opacity-80"
      />
    );
  };

  const renderManipulationLevel = (level, score) => {
    switch (level) {
      case "LOW":
        return (
          <div className="flex items-center">
            <div className="w-24 h-3 bg-gray-200 rounded-full">
              <div
                className="h-3 bg-green-500 rounded-full"
                style={{ width: `${score * 100}%` }}
              ></div>
            </div>
            <span className="ml-2 text-green-600">Low</span>
          </div>
        );
      case "MODERATE":
        return (
          <div className="flex items-center">
            <div className="w-24 h-3 bg-gray-200 rounded-full">
              <div
                className="h-3 bg-yellow-500 rounded-full"
                style={{ width: `${score * 100}%` }}
              ></div>
            </div>
            <span className="ml-2 text-yellow-600">Moderate</span>
          </div>
        );
      case "HIGH":
        return (
          <div className="flex items-center">
            <div className="w-24 h-3 bg-gray-200 rounded-full">
              <div
                className="h-3 bg-red-500 rounded-full"
                style={{ width: `${score * 100}%` }}
              ></div>
            </div>
            <span className="ml-2 text-red-600">High</span>
          </div>
        );
      default:
        return (
          <div className="flex items-center">
            <div className="w-24 h-3 bg-gray-200 rounded-full"></div>
            <span className="ml-2 text-gray-600">Unknown</span>
          </div>
        );
    }
  };

  const renderEntities = (entities) => {
    if (!entities || Object.keys(entities).length === 0) {
      return <p>No entities detected</p>;
    }

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        {Object.entries(entities).map(([entityType, entityList]) => (
          <div key={entityType} className="border rounded-md p-2">
            <span className="text-sm font-medium">{entityType}: </span>
            <span className="text-sm">{entityList.join(", ")}</span>
          </div>
        ))}
      </div>
    );
  };

  // Add this component inside your TruthLensApp component, but outside the return statement
  const ShareResultModal = ({ isOpen, onClose, result }) => {
    const [copied, setCopied] = useState(false);
    const shareUrl = window.location.href;
    const shareText = `TruthLens Analysis: "${result.claim}" was found to be ${result.verdict} with ${result.confidence}% confidence.`;

    const copyToClipboard = () => {
      navigator.clipboard.writeText(shareText + " " + shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    };

    if (!isOpen) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4 animate-fade-in">
        <div className="bg-white rounded-xl shadow-xl max-w-md w-full p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">Share Analysis Result</h3>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-6 w-6"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Share text
            </label>
            <div className="p-3 bg-gray-50 rounded-lg text-sm text-gray-800 border border-gray-200">
              {shareText}
            </div>
          </div>

          <div className="flex flex-wrap gap-2 mb-6">
            <button
              onClick={copyToClipboard}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
            >
              {copied ? (
                <>
                  <Check size={16} className="mr-2" />
                  Copied!
                </>
              ) : (
                <>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-4 w-4 mr-2"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                  Copy to Clipboard
                </>
              )}
            </button>
            <a
              href={`https://twitter.com/intent/tweet?text=${encodeURIComponent(
                shareText
              )}&url=${encodeURIComponent(shareUrl)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center px-4 py-2 bg-[#1DA1F2] text-white rounded-lg hover:bg-opacity-90 transition"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4 mr-2"
                fill="currentColor"
                viewBox="0 0 24 24"
              >
                <path d="M8.29 20.251c7.547 0 11.675-6.253 11.675-11.675 0-.178 0-.355-.012-.53A8.348 8.348 0 0022 5.92a8.19 8.19 0 01-2.357.646 4.118 4.118 0 001.804-2.27 8.224 8.224 0 01-2.605.996 4.107 4.107 0 00-6.993 3.743 11.65 11.65 0 01-8.457-4.287 4.106 4.106 0 001.27 5.477A4.072 4.072 0 012.8 9.713v.052a4.105 4.105 0 003.292 4.022 4.095 4.095 0 01-1.853.07 4.108 4.108 0 003.834 2.85A8.233 8.233 0 012 18.407a11.616 11.616 0 006.29 1.84" />
              </svg>
              Share on Twitter
            </a>
          </div>

          <button
            onClick={onClose}
            className="w-full py-2 text-gray-600 hover:text-gray-800 text-sm"
          >
            Close
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <ParticleBackground />
      <ParallaxHeader />

      {isPageLoading ? (
        <EnhancedLoadingAnimation />
      ) : (
        <div className="container mx-auto px-4 py-8">
          <div className="mb-6">
            <div className="flex border-b overflow-x-auto hide-scrollbar">
              <button
                className={`px-4 py-3 font-medium whitespace-nowrap ${
                  activeTab === "analysis"
                    ? "text-blue-600 border-b-2 border-blue-600"
                    : "text-gray-500 hover:text-gray-700"
                }`}
                onClick={() => setActiveTab("analysis")}
              >
                <Search size={16} className="inline mr-2" />
                Analysis Tool
              </button>
              <button
                className={`px-4 py-3 font-medium whitespace-nowrap ${
                  activeTab === "dashboard"
                    ? "text-blue-600 border-b-2 border-blue-600"
                    : "text-gray-500 hover:text-gray-700"
                }`}
                onClick={() => setActiveTab("dashboard")}
              >
                <BarChart2 size={16} className="inline mr-2" />
                Statistics Dashboard
              </button>
              <button
                className={`px-4 py-3 font-medium whitespace-nowrap ${
                  activeTab === "examples"
                    ? "text-blue-600 border-b-2 border-blue-600"
                    : "text-gray-500 hover:text-gray-700"
                }`}
                onClick={() => setActiveTab("examples")}
              >
                <AlertTriangle size={16} className="inline mr-2" />
                Fake News Examples
              </button>
              <button
                className={`px-4 py-3 font-medium whitespace-nowrap ${
                  activeTab === "demo"
                    ? "text-blue-600 border-b-2 border-blue-600"
                    : "text-gray-500 hover:text-gray-700"
                }`}
                onClick={() => setActiveTab("demo")}
              >
                <Bookmark size={16} className="inline mr-2" />
                Demo Samples
              </button>
              <button
                className={`px-4 py-3 font-medium whitespace-nowrap ${
                  activeTab === "about"
                    ? "text-blue-600 border-b-2 border-blue-600"
                    : "text-gray-500 hover:text-gray-700"
                }`}
                onClick={() => setActiveTab("about")}
              >
                <Info size={16} className="inline mr-2" />
                About
              </button>
            </div>
          </div>

          {activeTab === "analysis" && (
            <div className="grid grid-cols-1 gap-6">
              <div className="bg-white p-6 rounded-lg shadow-sm">
                <h2 className="text-xl font-semibold mb-4">Analyze Claim</h2>

                <div className="mb-4">
                  <label className="block text-gray-700 text-sm font-medium mb-2">
                    Enter a claim or URL to analyze
                  </label>
                  <div className="relative">
                    <textarea
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm"
                      rows="4"
                      value={claim}
                      onChange={handleClaimChange}
                      placeholder="Enter a claim to analyze, e.g., 'Scientists discovered that drinking coffee prevents cancer' or paste a URL"
                    />

                    {/* Add the prompt suggestion panel here */}
                    <PromptSuggestionPanel
                      prompt={claim}
                      suggestions={promptSuggestions}
                      onUseImproved={useImprovedPrompt}
                      isVisible={showPromptSuggestions}
                    />
                    {claim.trim().startsWith("http") && (
                      <div className="absolute top-2 right-2 bg-blue-100 text-blue-800 rounded-full px-2 py-1 text-xs flex items-center">
                        <Link size={12} className="mr-1" />
                        URL Detected
                      </div>
                    )}
                  </div>
                </div>

                <div className="mb-4 flex flex-wrap items-center gap-4">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="useRag"
                      checked={useRag}
                      onChange={(e) => setUseRag(e.target.checked)}
                      className="mr-2"
                    />
                    <label htmlFor="useRag" className="text-sm text-gray-700">
                      Use web search for context
                    </label>
                  </div>

                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="useKg"
                      checked={useKg}
                      onChange={(e) => setUseKg(e.target.checked)}
                      className="mr-2"
                    />
                    <label htmlFor="useKg" className="text-sm text-gray-700">
                      Use knowledge graph
                    </label>
                  </div>
                </div>

                {error && (
                  <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md">
                    {error}
                  </div>
                )}

                <button
                  className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50"
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                >
                  {isAnalyzing ? "Analyzing..." : "Analyze Claim"}
                </button>

                <div className="mt-4 text-sm text-gray-500">
                  <p>
                    Enter a claim above and click "Analyze Claim" to check its
                    veracity using Spark LLM and advanced analysis techniques.
                  </p>
                </div>
              </div>

              <div id="result-area" className="result-area">
                {isAnalyzing ? (
                  <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-full flex flex-col items-center justify-center animate-fade-in">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-4"></div>
                    <p className="text-gray-600 font-medium">
                      Analyzing with Spark LLM...
                    </p>
                    <p className="text-gray-500 text-sm mt-2">
                      Retrieving evidence and checking facts
                    </p>

                    <div className="w-48 h-1 bg-gray-200 rounded-full mt-6 overflow-hidden">
                      <div className="h-1 bg-blue-500 animate-progress-bar"></div>
                    </div>
                  </div>
                ) : result ? (
                  <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 overflow-auto max-h-[800px] animate-fade-in">
                    {/* Add this button next to the Share button */}
                    <div className="flex justify-between items-center mb-4">
                      <h2 className="text-xl font-semibold">
                        Analysis Results
                      </h2>
                      <div className="flex items-center space-x-3">
                        {result.verdict ? (
                          renderVerdictBadge(result.verdict)
                        ) : (
                          <div className="bg-gray-100 text-gray-800 rounded-full px-3 py-1 flex items-center">
                            <Info size={16} className="mr-1" />
                            Unverified
                          </div>
                        )}
                        <button
                          onClick={() => setShowShareModal(true)}
                          className="flex items-center text-blue-600 hover:text-blue-800 text-sm"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4 mr-1"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"
                            />
                          </svg>
                          Share
                        </button>
                        <button
                          onClick={generatePDF}
                          className="flex items-center text-blue-600 hover:text-blue-800 text-sm"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4 mr-1"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                            />
                          </svg>
                          Save PDF
                        </button>
                      </div>
                    </div>

                    {showResultsAnimation && (
                      <ResultsRevealAnimation
                        result={result}
                        isVisible={showResultsAnimation}
                        onComplete={() => setShowResultsAnimation(false)}
                      />
                    )}

                    {/* Add confidence gauge chart */}
                    <div className="mb-6">
                      <div className="text-sm text-gray-500 mb-1">
                        Confidence
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Standard progress bar (keep this) */}
                        <div className="flex items-center">
                          <div className="w-full h-3 bg-gray-200 rounded-full">
                            <div
                              className={`h-3 rounded-full ${
                                result.verdict === "FAKE" ||
                                result.verdict === "False"
                                  ? "bg-red-500"
                                  : result.verdict === "MISLEADING" ||
                                    result.verdict === "Partially True"
                                  ? "bg-yellow-500"
                                  : result.verdict === "REAL" ||
                                    result.verdict === "True"
                                  ? "bg-green-500"
                                  : "bg-gray-500"
                              }`}
                              style={{ width: `${result.confidence || 0}%` }}
                            ></div>
                          </div>
                          <span className="ml-2 font-medium">
                            {result.confidence || 0}%
                          </span>
                        </div>

                        {/* Add semicircular gauge chart */}
                        <div className="h-32">
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                startAngle={180}
                                endAngle={0}
                                data={[
                                  {
                                    name: "Confidence",
                                    value: result.confidence || 0,
                                  },
                                  {
                                    name: "Remaining",
                                    value: 100 - (result.confidence || 0),
                                  },
                                ]}
                                cx="50%"
                                cy="100%"
                                outerRadius={80}
                                innerRadius={60}
                                dataKey="value"
                              >
                                <Cell
                                  key={`cell-0`}
                                  fill={
                                    result.verdict === "FAKE" ||
                                    result.verdict === "False"
                                      ? "#ef4444"
                                      : result.verdict === "MISLEADING" ||
                                        result.verdict === "Partially True"
                                      ? "#f59e0b"
                                      : result.verdict === "REAL" ||
                                        result.verdict === "True"
                                      ? "#10b981"
                                      : "#9ca3af"
                                  }
                                />
                                <Cell key={`cell-1`} fill="#e5e7eb" />
                              </Pie>
                              <text
                                x="50%"
                                y="90%"
                                textAnchor="middle"
                                dominantBaseline="middle"
                                className="font-bold"
                                fontSize="24"
                              >
                                {result.confidence || 0}%
                              </text>
                            </PieChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>
                    <div className="mb-6">
                      <div className="text-sm text-gray-500 mb-1">
                        Confidence Factors
                      </div>
                      <div className="p-4 bg-gray-50 rounded-lg">
                        <div className="h-48">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              layout="vertical"
                              data={[
                                {
                                  name: "Source Credibility",
                                  score: result.credibility_assessment
                                    ? result.credibility_assessment.score * 100
                                    : 65,
                                },
                                {
                                  name: "Evidence Strength",
                                  score: result.confidence * 0.85, // Estimated from overall confidence
                                },
                                {
                                  name: "Factual Consistency",
                                  score: result.confidence * 0.9, // Estimated from overall confidence
                                },
                                {
                                  name: "Information Recency",
                                  score: 75, // Example value
                                },
                              ]}
                              margin={{
                                top: 5,
                                right: 30,
                                left: 120,
                                bottom: 5,
                              }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis type="number" domain={[0, 100]} />
                              <YAxis dataKey="name" type="category" />
                              <Tooltip
                                formatter={(value) => [
                                  `${value.toFixed(1)}%`,
                                  "Score",
                                ]}
                              />
                              <Bar
                                dataKey="score"
                                fill={
                                  result.verdict === "FAKE" ||
                                  result.verdict === "False"
                                    ? "#ef4444"
                                    : result.verdict === "MISLEADING" ||
                                      result.verdict === "Partially True"
                                    ? "#f59e0b"
                                    : "#10b981"
                                }
                              />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </div>

                    {/* Replace the existing verdict visualization with our new enhanced component */}
                    <EnhancedVerdictAssessment result={result} />
                    {/* Add the confidence factors widget after the verdict assessment */}
                    {result.confidence_factors && (
                      <ConfidenceFactorsWidget
                        factors={result.confidence_factors}
                      />
                    )}
                    {/* Add this Spark LLM badge */}
                    <div className="mb-4 flex justify-end">
                      <div className="bg-indigo-100 text-indigo-800 text-xs px-2 py-1 rounded flex items-center">
                        <Coffee size={12} className="mr-1" />
                        Powered by Spark LLM
                      </div>
                    </div>

                    <EmotionalAnalysisChart
                      sentimentAnalysis={result.emotional_manipulation}
                    />

                    {result.verification_analysis && (
                      <VerificationBadge verificationPerformed={true} />
                    )}

                    {result.is_url_input && (
                      <>
                        {renderSourceMetadata(result.source_metadata)}
                        {renderCredibilityAssessment(
                          result.credibility_assessment
                        )}
                      </>
                    )}

                    <div className="my-4 border-t border-gray-200 pt-4">
                      <div className="text-sm text-gray-500 mb-1">
                        Emotional Manipulation
                      </div>
                      {renderManipulationLevel(
                        result.emotional_manipulation.level,
                        result.emotional_manipulation.score
                      )}
                    </div>
                    <div className="mb-4">
                      <div className="text-sm text-gray-500 mb-1">
                        Emotional Analysis
                      </div>
                      <p className="text-gray-800 bg-gray-50 p-3 rounded-lg">
                        {result.emotional_manipulation.explanation}
                      </p>
                    </div>
                    {result.title_content_contradiction && (
                      <TitleContentContradictionWidget
                        contradiction={result.title_content_contradiction}
                      />
                    )}

                    {result.trust_lens_score !== undefined &&
                      result.trust_lens_score !== null && (
                        <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                          <div className="text-sm text-gray-500 mb-1">
                            Trust Lens Score
                          </div>
                          <div className="flex items-center justify-center">
                            <TrustBadge3D score={result.trust_lens_score} />
                          </div>
                          <p className="text-xs text-gray-600 mt-2 text-center">
                            A comprehensive score combining source credibility,
                            factual accuracy, and content neutrality.
                          </p>
                        </div>
                      )}
                    {/* Add propaganda techniques display */}
                    {result.emotional_manipulation &&
                      result.emotional_manipulation.propaganda_analysis &&
                      result.emotional_manipulation.propaganda_analysis
                        .has_propaganda && (
                        <div className="mt-3 p-3 bg-yellow-50 rounded-lg">
                          <h4 className="font-medium mb-2">
                            Detected Propaganda Techniques
                          </h4>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                            {Object.entries(
                              result.emotional_manipulation.propaganda_analysis
                                .detected_techniques
                            ).map(([technique, data]) => (
                              <div
                                key={technique}
                                className="bg-white p-2 rounded border border-yellow-200"
                              >
                                <div className="font-medium">
                                  {technique
                                    .replace("_", " ")
                                    .replace(/\b\w/g, (l) => l.toUpperCase())}
                                </div>
                                <div className="text-xs text-gray-600 mt-1">
                                  Examples: {data.examples.join(", ")}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    <div className="mb-4">
                      <div className="text-sm text-gray-500 mb-1">
                        Detected Entities
                      </div>
                      {renderEntities(result.entities)}
                    </div>
                    {result && (
                      <>
                        {/* Debug information */}
                        <div className="hidden">
                          Debug: AI Detection available:{" "}
                          {result.ai_detection ? "Yes" : "No"}
                        </div>

                        {/* Enhanced AI Detection Widget with better prop handling */}
                        <InteractiveAIDetectionBadge
                          aiDetection={result.ai_detection || null}
                        />
                      </>
                    )}
                    <div className="text-xs text-gray-500 mt-6 pt-4 border-t border-gray-200 flex items-center justify-end">
                      <Clock size={14} className="mr-1" />
                      Analysis completed in{" "}
                      {result.processing_time?.toFixed(2) || "?"} seconds
                    </div>
                  </div>
                ) : (
                  <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 h-full animate-fade-in">
                    <div className="flex flex-col items-center justify-center h-full text-center">
                      <Search className="text-gray-400 mb-4" size={48} />
                      <h3 className="text-lg font-medium text-gray-700 mb-2">
                        No Analysis Yet
                      </h3>
                      <p className="text-gray-500">
                        Enter a claim and click "Analyze Claim" to get started.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "dashboard" && (
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold mb-6 flex items-center">
                <BarChart2 size={20} className="mr-2 text-blue-600" />
                Fake News Statistics Dashboard
              </h2>

              <div className="mb-6">
                <div className="flex border-b mb-4">
                  <button
                    className={`px-4 py-2 font-medium ${
                      activeDashboardTab === "trends"
                        ? "text-blue-600 border-b-2 border-blue-600"
                        : "text-gray-500"
                    }`}
                    onClick={() => setActiveDashboardTab("trends")}
                  >
                    Yearly Trends
                  </button>
                  <button
                    className={`px-4 py-2 font-medium ${
                      activeDashboardTab === "categories"
                        ? "text-blue-600 border-b-2 border-blue-600"
                        : "text-gray-500"
                    }`}
                    onClick={() => setActiveDashboardTab("categories")}
                  >
                    Categories
                  </button>
                  <button
                    className={`px-4 py-2 font-medium ${
                      activeDashboardTab === "sources"
                        ? "text-blue-600 border-b-2 border-blue-600"
                        : "text-gray-500"
                    }`}
                    onClick={() => setActiveDashboardTab("sources")}
                  >
                    Source Analysis
                  </button>
                </div>

                {activeDashboardTab === "trends" && (
                  <div>
                    <div className="p-4 bg-gray-50 rounded-lg mb-4">
                      <h3 className="text-lg font-medium mb-2">
                        Fake News Trends (2024)
                      </h3>
                      <p className="text-gray-600 mb-4">
                        Monthly breakdown of fact-checked content across
                        categories
                      </p>

                      <div className="h-80">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={fakeNewsStats}
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="month" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line
                              type="monotone"
                              dataKey="fakeNews"
                              name="Fake News"
                              stroke="#f87171"
                              activeDot={{ r: 8 }}
                              strokeWidth={2}
                            />
                            <Line
                              type="monotone"
                              dataKey="misleading"
                              name="Misleading"
                              stroke="#fbbf24"
                              strokeWidth={2}
                            />
                            <Line
                              type="monotone"
                              dataKey="credible"
                              name="Credible"
                              stroke="#34d399"
                              strokeWidth={2}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="bg-red-50 rounded-lg p-4 border border-red-100">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-medium text-red-800">
                            Fake News
                          </h3>
                          <AlertCircle size={18} className="text-red-500" />
                        </div>
                        <div className="text-2xl font-bold text-red-600 mb-1">
                          6,468
                        </div>
                        <div className="text-sm text-red-700 flex items-center">
                          <TrendingUp size={16} className="mr-1" />
                          12.4% increase from last year
                        </div>
                      </div>

                      <div className="bg-yellow-50 rounded-lg p-4 border border-yellow-100">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-medium text-yellow-800">
                            Misleading Content
                          </h3>
                          <AlertTriangle
                            size={18}
                            className="text-yellow-500"
                          />
                        </div>
                        <div className="text-2xl font-bold text-yellow-600 mb-1">
                          4,269
                        </div>
                        <div className="text-sm text-yellow-700 flex items-center">
                          <TrendingUp size={16} className="mr-1" />
                          8.7% increase from last year
                        </div>
                      </div>

                      <div className="bg-green-50 rounded-lg p-4 border border-green-100">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-medium text-green-800">
                            Credible Content
                          </h3>
                          <Check size={18} className="text-green-500" />
                        </div>
                        <div className="text-2xl font-bold text-green-600 mb-1">
                          10,115
                        </div>
                        <div className="text-sm text-green-700 flex items-center">
                          <TrendingUp size={16} className="mr-1" />
                          3.2% increase from last year
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {activeDashboardTab === "categories" && (
                  <div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="p-4 bg-gray-50 rounded-lg">
                        <h3 className="text-lg font-medium mb-2">
                          Fake News by Category
                        </h3>
                        <p className="text-gray-600 mb-4">
                          Distribution of misleading content across topics
                        </p>

                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                data={categoryBreakdown}
                                cx="50%"
                                cy="50%"
                                labelLine={false}
                                label={({ name, percent }) =>
                                  `${name}: ${(percent * 100).toFixed(0)}%`
                                }
                                outerRadius={80}
                                fill="#8884d8"
                                dataKey="value"
                              >
                                {categoryBreakdown.map((entry, index) => (
                                  <Cell
                                    key={`cell-${index}`}
                                    fill={COLORS[index % COLORS.length]}
                                  />
                                ))}
                              </Pie>
                              <Tooltip />
                              <Legend />
                            </PieChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      <div>
                        <div className="p-4 bg-gray-50 rounded-lg mb-4">
                          <h3 className="text-lg font-medium mb-2">
                            Top Categories
                          </h3>
                          <p className="text-gray-600 mb-4">
                            Categories with highest fake news prevalence
                          </p>

                          <div className="space-y-3">
                            <div>
                              <div className="flex justify-between mb-1">
                                <span className="text-sm font-medium text-gray-700">
                                  Health
                                </span>
                                <span className="text-sm text-gray-600">
                                  32%
                                </span>
                              </div>
                              <div className="w-full h-2 bg-gray-200 rounded-full">
                                <div
                                  className="h-2 bg-red-500 rounded-full"
                                  style={{ width: "32%" }}
                                ></div>
                              </div>
                            </div>

                            <div>
                              <div className="flex justify-between mb-1">
                                <span className="text-sm font-medium text-gray-700">
                                  Politics
                                </span>
                                <span className="text-sm text-gray-600">
                                  28%
                                </span>
                              </div>
                              <div className="w-full h-2 bg-gray-200 rounded-full">
                                <div
                                  className="h-2 bg-blue-500 rounded-full"
                                  style={{ width: "28%" }}
                                ></div>
                              </div>
                            </div>

                            <div>
                              <div className="flex justify-between mb-1">
                                <span className="text-sm font-medium text-gray-700">
                                  Celebrity
                                </span>
                                <span className="text-sm text-gray-600">
                                  15%
                                </span>
                              </div>
                              <div className="w-full h-2 bg-gray-200 rounded-full">
                                <div
                                  className="h-2 bg-purple-500 rounded-full"
                                  style={{ width: "15%" }}
                                ></div>
                              </div>
                            </div>

                            <div>
                              <div className="flex justify-between mb-1">
                                <span className="text-sm font-medium text-gray-700">
                                  Science
                                </span>
                                <span className="text-sm text-gray-600">
                                  12%
                                </span>
                              </div>
                              <div className="w-full h-2 bg-gray-200 rounded-full">
                                <div
                                  className="h-2 bg-green-500 rounded-full"
                                  style={{ width: "12%" }}
                                ></div>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="p-4 bg-yellow-50 rounded-lg border border-yellow-100">
                          <div className="flex items-center mb-2">
                            <AlertTriangle
                              size={18}
                              className="text-yellow-600 mr-2"
                            />
                            <h3 className="font-medium text-yellow-800">
                              Key Insight
                            </h3>
                          </div>
                          <p className="text-sm text-yellow-700">
                            Health misinformation continues to dominate fake
                            news content in 2024, followed closely by political
                            claims. This trend has remained consistent over the
                            last 3 years.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {activeDashboardTab === "sources" && (
                  <div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="p-4 bg-gray-50 rounded-lg">
                        <h3 className="text-lg font-medium mb-2">
                          Fake News by Source Type
                        </h3>
                        <p className="text-gray-600 mb-4">
                          Where misinformation originates from most frequently
                        </p>

                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={credibilitySourceData}
                              margin={{
                                top: 5,
                                right: 30,
                                left: 20,
                                bottom: 5,
                              }}
                            >
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="name" />
                              <YAxis />
                              <Tooltip />
                              <Bar
                                dataKey="value"
                                name="Percentage"
                                fill="#8884d8"
                              >
                                {credibilitySourceData.map((entry, index) => (
                                  <Cell
                                    key={`cell-${index}`}
                                    fill={COLORS[index % COLORS.length]}
                                  />
                                ))}
                              </Bar>
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      <div>
                        <div className="p-4 bg-gray-50 rounded-lg mb-4">
                          <h3 className="text-lg font-medium mb-2">
                            Credibility Factors
                          </h3>
                          <p className="text-gray-600 mb-4">
                            Common patterns in low-credibility sources
                          </p>

                          <div className="space-y-4">
                            <div className="flex items-start">
                              <div className="bg-red-100 p-2 rounded-full mr-3 mt-1">
                                <AlertCircle
                                  size={16}
                                  className="text-red-600"
                                />
                              </div>
                              <div>
                                <h4 className="font-medium text-gray-800">
                                  Anonymous Authors
                                </h4>
                                <p className="text-sm text-gray-600">
                                  73% of fake news articles lack clear author
                                  attribution or credentials
                                </p>
                              </div>
                            </div>

                            <div className="flex items-start">
                              <div className="bg-red-100 p-2 rounded-full mr-3 mt-1">
                                <AlertCircle
                                  size={16}
                                  className="text-red-600"
                                />
                              </div>
                              <div>
                                <h4 className="font-medium text-gray-800">
                                  Domain Age
                                </h4>
                                <p className="text-sm text-gray-600">
                                  68% of fake news websites were created within
                                  the last 2 years
                                </p>
                              </div>
                            </div>

                            <div className="flex items-start">
                              <div className="bg-red-100 p-2 rounded-full mr-3 mt-1">
                                <AlertCircle
                                  size={16}
                                  className="text-red-600"
                                />
                              </div>
                              <div>
                                <h4 className="font-medium text-gray-800">
                                  Emotional Language
                                </h4>
                                <p className="text-sm text-gray-600">
                                  89% use highly emotional language and
                                  sensationalist headlines
                                </p>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
                          <div className="flex items-center mb-2">
                            <Info size={18} className="text-blue-600 mr-2" />
                            <h3 className="font-medium text-blue-800">
                              Credibility Tip
                            </h3>
                          </div>
                          <p className="text-sm text-blue-700">
                            Always check the author credentials, publication
                            date, and domain reliability before trusting an
                            article. Anonymous sources and very recent domains
                            are major red flags.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === "examples" && (
            <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <AlertTriangle size={20} className="mr-2 text-red-500" />
                Examples of Fake News
              </h2>
              <p className="text-gray-600 mb-6">
                These are examples of debunked fake news stories that have
                circulated online. Learn to recognize common patterns.
              </p>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {fakeNewsExamples.map((example, index) => (
                  <div
                    key={index}
                    className="border rounded-lg overflow-hidden bg-white shadow-sm hover:shadow-md transition"
                  >
                    <div className="relative">
                      <img
                        src={example.image}
                        alt={example.title}
                        className="w-full h-48 object-cover"
                      />
                      <div className="absolute top-0 right-0 m-2">
                        {renderVerdictBadge("FAKE")}
                      </div>
                    </div>

                    <div className="p-4">
                      <h3 className="font-medium text-lg mb-2">
                        {example.title}
                      </h3>
                      <p className="text-gray-600 text-sm mb-4">
                        {example.description}
                      </p>

                      <div className="space-y-2 text-sm text-gray-500">
                        <div className="flex items-center">
                          <Globe size={14} className="mr-2 text-gray-400" />
                          Source:{" "}
                          <span className="ml-1 text-red-500">
                            {example.source}
                          </span>
                        </div>
                        <div className="flex items-center">
                          <User size={14} className="mr-2 text-gray-400" />
                          Author: {example.author}
                        </div>
                        <div className="flex items-center">
                          <Calendar size={14} className="mr-2 text-gray-400" />
                          Published: {example.date}
                        </div>
                        <div className="flex items-center">
                          <Shield size={14} className="mr-2 text-gray-400" />
                          Credibility Score:{" "}
                          <span className="ml-1 text-red-500 font-medium">
                            {example.credibilityScore}
                          </span>
                        </div>
                      </div>

                      <div className="mt-4 pt-3 border-t border-gray-100">
                        <h4 className="font-medium text-sm mb-2">Red Flags:</h4>
                        <ul className="text-xs text-gray-600 space-y-1">
                          <li className="flex items-center">
                            <ChevronRight
                              size={12}
                              className="mr-1 text-red-500"
                            />
                            Emotional manipulation
                          </li>
                          <li className="flex items-center">
                            <ChevronRight
                              size={12}
                              className="mr-1 text-red-500"
                            />
                            Unverifiable claims
                          </li>
                          <li className="flex items-center">
                            <ChevronRight
                              size={12}
                              className="mr-1 text-red-500"
                            />
                            Non-credible source
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-8 p-4 bg-blue-50 rounded-lg border border-blue-100">
                <h3 className="font-medium text-lg text-blue-800 mb-2 flex items-center">
                  <Shield size={20} className="mr-2 text-blue-600" />
                  How to Spot Fake News
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                  <div className="bg-white p-3 rounded-lg border border-blue-100">
                    <h4 className="font-medium text-blue-900 mb-2">
                      Check the Source
                    </h4>
                    <p className="text-sm text-gray-600">
                      Investigate the website's About page, look for contact
                      information, and verify if it's a reputable news outlet.
                    </p>
                  </div>
                  <div className="bg-white p-3 rounded-lg border border-blue-100">
                    <h4 className="font-medium text-blue-900 mb-2">
                      Verify the Author
                    </h4>
                    <p className="text-sm text-gray-600">
                      Look for the author's credentials, other publications, and
                      professional background.
                    </p>
                  </div>
                  <div className="bg-white p-3 rounded-lg border border-blue-100">
                    <h4 className="font-medium text-blue-900 mb-2">
                      Cross-Check Claims
                    </h4>
                    <p className="text-sm text-gray-600">
                      Check if multiple reputable sources are reporting the same
                      information with similar facts.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === "demo" && (
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h2 className="text-xl font-semibold mb-4">Demo Samples</h2>
              <p className="text-gray-600 mb-6">
                Click on any of these examples to load them into the analyzer.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {demoSamples.map((sample, index) => (
                  <div
                    key={index}
                    className="border rounded-lg p-4 cursor-pointer hover:bg-gray-50 transition"
                    onClick={() => {
                      loadSample(sample);
                      setActiveTab("analysis");
                    }}
                  >
                    <h3 className="font-medium mb-2">{sample.title}</h3>
                    <p className="text-gray-600 text-sm line-clamp-3">
                      {sample.claim}
                    </p>
                    <div className="mt-2 text-blue-600 text-sm">
                      Click to analyze
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === "about" && (
            <div className="bg-white p-6 rounded-lg shadow-sm">
              <h2 className="text-xl font-semibold mb-4">About TruthLens</h2>

              <div className="mb-6">
                <h3 className="text-lg font-medium mb-2">What is TruthLens?</h3>
                <p className="text-gray-700 mb-4">
                  TruthLens is an advanced fake news detection system powered by
                  Spark LLM and enhanced with Retrieval Augmented Generation
                  (RAG), emotional manipulation analysis, and knowledge graph
                  verification. It's designed to help users identify potentially
                  misleading or false information online.
                </p>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-medium mb-2">How It Works</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="border rounded-lg p-4">
                    <div className="flex items-center mb-2">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 mr-2">
                        1
                      </div>
                      <h4 className="font-medium">Text Analysis</h4>
                    </div>
                    <p className="text-gray-600 text-sm">
                      Spark LLM analyzes the text for claims, language patterns,
                      and indicators of misinformation.
                    </p>
                  </div>

                  <div className="border rounded-lg p-4">
                    <div className="flex items-center mb-2">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 mr-2">
                        2
                      </div>
                      <h4 className="font-medium">Fact Verification</h4>
                    </div>
                    <p className="text-gray-600 text-sm">
                      The RAG system retrieves relevant contextual information
                      from reliable sources to verify claims.
                    </p>
                  </div>

                  <div className="border rounded-lg p-4">
                    <div className="flex items-center mb-2">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 mr-2">
                        3
                      </div>
                      <h4 className="font-medium">Entity Verification</h4>
                    </div>
                    <p className="text-gray-600 text-sm">
                      The system extracts entities and uses a knowledge graph to
                      verify their relationships.
                    </p>
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-medium mb-2">Technology Stack</h3>
                <ul className="list-disc pl-5 text-gray-700">
                  <li className="mb-1">
                    <strong>Spark LLM</strong> for advanced natural language
                    understanding and fact checking
                  </li>
                  <li className="mb-1">
                    Retrieval Augmented Generation (RAG) for context-aware
                    fact-checking
                  </li>
                  <li className="mb-1">
                    FAISS for efficient vector similarity search
                  </li>
                  <li className="mb-1">
                    Sentiment analysis and emotion detection systems
                  </li>
                  <li className="mb-1">
                    Named Entity Recognition (NER) for extracting key entities
                  </li>
                  <li className="mb-1">
                    Knowledge Graph (Neo4j) for entity relationship verification
                  </li>
                  <li className="mb-1">FastAPI backend and React frontend</li>
                </ul>
              </div>

              <div>
                <h3 className="text-lg font-medium mb-2">Limitations</h3>
                <p className="text-gray-700">
                  While TruthLens is a powerful tool for detecting potential
                  misinformation, it's not perfect. Always use critical thinking
                  and consult multiple sources when evaluating information
                  online. The system works best with English-language content
                  and may have limitations with highly technical or specialized
                  content.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      <footer className="bg-gray-800 text-white py-8 mt-12">
        <div className="container mx-auto px-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="mb-6 md:mb-0">
              <div className="flex items-center">
                <Droplet className="mr-2" size={24} />
                <span className="font-semibold text-xl">TruthLens</span>
              </div>
              <div className="text-sm text-gray-400 mt-2">
                Advanced Fake News Detection System
              </div>
              <div className="mt-4 flex space-x-4">
                <a
                  href="#"
                  className="text-gray-400 hover:text-white transition"
                >
                  <span className="sr-only">GitHub</span>
                  <svg
                    className="h-6 w-6"
                    fill="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      fillRule="evenodd"
                      d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z"
                      clipRule="evenodd"
                    />
                  </svg>
                </a>
              </div>
            </div>
            <div className="text-sm text-gray-400">
              © 2025 TruthLens Project. All rights reserved.
            </div>
          </div>
        </div>
      </footer>

      {result && (
        <ShareResultModal
          isOpen={showShareModal}
          onClose={() => setShowShareModal(false)}
          result={result}
        />
      )}
      {/* Add this at the end of your return statement, before the closing </div> */}
      <HistoryPanel
        isVisible={showHistory}
        onClose={() => setShowHistory(false)}
        history={analysisHistory}
        onSelectItem={handleHistoryItemSelect}
      />
    </div>
  );
};

export default TruthLensApp;
