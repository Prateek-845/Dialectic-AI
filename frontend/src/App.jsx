import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './index.css';

function App() {
  const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [user, setUser] = useState(localStorage.getItem('username') || null);

  const [isLogin, setIsLogin] = useState(true);
  const [authUsername, setAuthUsername] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authError, setAuthError] = useState('');

  const [article, setArticle] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [showHistory, setShowHistory] = useState(false);
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);


  const [threadId, setThreadId] = useState(null);
  const [juryFeedback, setJuryFeedback] = useState('');
  const [isPaused, setIsPaused] = useState(false);
  const resultsContainerRef = useRef(null);


  const handleAuth = async (e) => {
    e.preventDefault();
    setAuthError('');
    try {
      const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
      const res = await axios.post(`${API_URL}${endpoint}`, {
        username: authUsername,
        password: authPassword
      });
      setToken(res.data.token);
      setUser(res.data.username);
      localStorage.setItem('token', res.data.token);
      localStorage.setItem('username', res.data.username);
    } catch (err) {
      setAuthError(err.response?.data?.error || 'Authentication failed');
    }
  };

  const handleLogout = () => {
    setToken(null);
    setUser(null);
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    setResult(null);
  };

  const loadHistory = async () => {
    if (!token) return;
    setHistoryLoading(true);
    try {
      const res = await axios.get(`${API_URL}/api/debates`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setHistory(res.data);
    } catch (err) {
      console.error(err);
      if (err.response?.status === 401 || err.response?.status === 403) handleLogout();
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    if (showHistory) {
      loadHistory();
    }
  }, [showHistory]);

  const handleAnalyze = async (resumeAction = null) => {
    if (!article && !resumeAction) return;

    let currentThreadId = threadId;
    if (!resumeAction) {
      currentThreadId = crypto.randomUUID();
      setThreadId(currentThreadId);
      setResult(null);
      setJuryFeedback('');
    }

    setLoading(true);
    setIsPaused(false);
    setError(null);

    try {
      const payload = {
        article,
        thread_id: currentThreadId,
        action: resumeAction,
        jury_feedback: juryFeedback
      };

      const response = await fetch(`${API_URL}/api/debates/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        if (response.status === 401 || response.status === 403) {
          handleLogout();
          throw new Error('Session expired. Please log in again.');
        }
        try {
          const errData = await response.json();
          throw new Error(errData.error || 'Failed to start debate stream');
        } catch(e) {
          throw new Error(e.message || 'Failed to start debate stream');
        }
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let latestState = null;

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          setLoading(false);
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.substring(6);
            if (dataStr === '[DONE]') {
              setLoading(false);
              if (latestState && !latestState.final_summary) {
                setIsPaused(true);
              }
              break;
            }
            try {
              const stateUpdate = JSON.parse(dataStr);
              if (stateUpdate.error) {
                setError("AI Engine Error: " + stateUpdate.error);
                setLoading(false);
                break;
              }
              latestState = stateUpdate;
              setResult(stateUpdate);
            } catch (e) {
              console.error("Parse error on stream chunk:", e);
            }
          }
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to process debate');
      setLoading(false);
    }
  };

  const exportPDF = () => {
    window.print();
  };

  if (!token) {
    return (
      <div className="min-h-screen bg-bg-primary flex flex-col items-center justify-center p-4">
        <div className="w-full max-w-sm mb-8 text-center">
          <h1 className="text-2xl font-semibold tracking-tight text-white mb-2">Dialectic AI</h1>
          <p className="text-sm text-gray-400">Sign in to your account</p>
        </div>

        <div className="bg-card-bg p-8 rounded-xl border border-white/5 shadow-2xl w-full max-w-sm">
          <form onSubmit={handleAuth} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1.5">Username</label>
              <input
                type="text" required
                value={authUsername} onChange={e => setAuthUsername(e.target.value)}
                className="w-full bg-black/20 border border-white/10 rounded-lg p-2.5 text-sm text-white focus:border-accent-blue focus:ring-1 focus:ring-accent-blue outline-none transition-all"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1.5">Password</label>
              <input
                type="password" required minLength={6}
                value={authPassword} onChange={e => setAuthPassword(e.target.value)}
                className="w-full bg-black/20 border border-white/10 rounded-lg p-2.5 text-sm text-white focus:border-accent-blue focus:ring-1 focus:ring-accent-blue outline-none transition-all"
              />
            </div>

            {authError && <p className="text-rose-400 text-xs bg-rose-500/10 p-2.5 rounded text-center border border-rose-500/20">{authError}</p>}

            <button type="submit" className="w-full bg-white text-black font-medium py-2.5 mt-2 rounded-lg hover:bg-gray-200 transition-colors text-sm">
              {isLogin ? 'Sign In' : 'Create Account'}
            </button>
          </form>

          <div className="mt-6 text-center">
            <button onClick={() => { setIsLogin(!isLogin); setAuthError(''); }} className="text-xs text-gray-400 hover:text-white transition-colors">
              {isLogin ? "Don't have an account? Sign up" : "Already have an account? Sign in"}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg-primary font-sans">
      <div className="max-w-[1400px] mx-auto px-6 py-8 flex gap-8">


        {showHistory && (
          <aside className="w-72 flex-shrink-0 border-r border-white/5 pr-6 h-[calc(100vh-4rem)] overflow-y-auto sticky top-8 no-print">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-sm font-semibold tracking-wide text-gray-300">History</h2>
              <button onClick={() => setShowHistory(false)} className="text-gray-500 hover:text-white transition-colors">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
              </button>
            </div>

            {historyLoading ? (
              <div className="flex justify-center py-8"><div className="w-4 h-4 border-2 border-accent-blue border-t-transparent rounded-full animate-spin"></div></div>
            ) : history.length === 0 ? (
              <p className="text-gray-500 text-xs">No past debates found.</p>
            ) : (
              <div className="space-y-3">
                {history.map((h) => (
                  <div
                    key={h._id}
                    onClick={() => { setResult(h); setShowHistory(false); setArticle(h.original_article || ''); }}
                    className="p-3 bg-card-bg rounded-lg border border-white/5 cursor-pointer hover:bg-white/5 hover:border-white/10 transition-all group"
                  >
                    <p className="text-xs text-gray-400 line-clamp-2 mb-2 group-hover:text-gray-200 transition-colors">{h.original_article}</p>
                    <div className="flex justify-between items-center text-[10px] text-gray-500">
                      <span>{new Date(h.createdAt).toLocaleDateString()}</span>
                      {h.synthesis_neutral && <span>Pol: {h.synthesis_neutral.synthesis_polarity.toFixed(2)}</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </aside>
        )}


        <main className="flex-1 print-w-full">

          <header className="flex justify-between items-center mb-10 pb-6 border-b border-white/5 no-print">
            <h1 className="text-lg font-semibold tracking-tight text-white flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-white"></div>
              Dialectic AI
            </h1>
            <div className="flex items-center gap-4 text-sm">
              <span className="text-gray-400">Hi, <span className="text-gray-200">{user}</span></span>
              <button onClick={() => setShowHistory(!showHistory)} className="text-gray-400 hover:text-white transition-colors">History</button>
              <button onClick={handleLogout} className="text-gray-400 hover:text-rose-400 transition-colors">Sign Out</button>
            </div>
          </header>


          {!result && !loading && (
            <div className="max-w-2xl mx-auto mt-12 animate-fade-in no-print">
              <h2 className="text-2xl font-semibold mb-2 text-white tracking-tight">New Analysis</h2>
              <p className="text-sm text-gray-400 mb-6">Enter an article or URL to synthesize multiple perspectives.</p>

              <div className="bg-card-bg p-1.5 rounded-xl border border-white/5 shadow-sm focus-within:border-white/20 focus-within:ring-1 focus-within:ring-white/20 transition-all">
                <textarea
                  value={article}
                  onChange={(e) => setArticle(e.target.value)}
                  placeholder="Paste article text or URL..."
                  className="w-full h-32 bg-transparent p-3 text-sm text-gray-200 resize-none focus:outline-none"
                />
                <div className="flex justify-end p-2 border-t border-white/5">
                  <button
                    onClick={() => handleAnalyze(null)}
                    disabled={!article.trim()}
                    className="bg-white text-black font-medium px-4 py-1.5 rounded-lg text-sm hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Analyze
                  </button>
                </div>
              </div>
              {error && <p className="text-rose-400 text-sm mt-4 p-3 bg-rose-500/10 border border-rose-500/20 rounded-lg">{error}</p>}
            </div>
          )}


          {loading && !result && (
            <div className="flex flex-col items-center justify-center py-32 animate-pulse no-print">
              <div className="w-8 h-8 border-2 border-white/10 border-t-white rounded-full animate-spin mb-4"></div>
              <p className="text-sm font-medium text-gray-400 tracking-wide">Initializing agents...</p>
            </div>
          )}


          {result && (
            <div className="animate-fade-in print:bg-transparent" ref={resultsContainerRef}>


              <div className="flex justify-between items-center mb-8 no-print">
                <div className="flex items-center gap-3">
                  <button onClick={() => { setResult(null); setLoading(false); setArticle(''); setIsPaused(false); setThreadId(null); setJuryFeedback(''); }} className="text-sm font-medium text-gray-400 hover:text-white transition-colors flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
                    New Analysis
                  </button>
                </div>
                <div className="flex items-center gap-4">
                  {loading && (
                    <span className="flex items-center text-xs font-medium text-emerald-400 animate-pulse">
                      <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full mr-2"></span> Streaming
                    </span>
                  )}
                  {result.final_summary && (
                    <button onClick={exportPDF} className="text-sm font-medium text-gray-400 hover:text-white transition-colors flex items-center gap-2 px-3 py-1.5 rounded-md border border-white/10 hover:bg-white/5">
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                      Export
                    </button>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-[1fr_350px] gap-8">


                <div className="space-y-6">
                  <h3 className="text-sm font-semibold text-gray-300 tracking-wide uppercase">Argument Trace</h3>

                  <div className="flex gap-4 mb-2">
                    <span className="flex items-center text-xs text-gray-500"><span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5"></span>Verified Entity</span>
                    <span className="flex items-center text-xs text-gray-500"><span className="w-1.5 h-1.5 rounded-full bg-blue-400 mr-1.5"></span>Web Source</span>
                    <span className="flex items-center text-xs text-gray-500"><span className="w-1.5 h-1.5 rounded-full bg-rose-400 mr-1.5"></span>Unverified</span>
                  </div>

                  <div className="space-y-4">
                    {result.debate_log && result.debate_log.map((round, idx) => (
                      <div key={idx} className="bg-card-bg rounded-xl border border-white/5 p-5">
                        <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest mb-4">Round {round.iteration}</div>


                        <div className="mb-6">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-rose-500/80"></span>
                              <strong className="text-gray-200 text-sm font-medium">{result.persona_a}</strong>
                            </div>
                            <span className="text-xs font-mono text-gray-500">{round.a_score}</span>
                          </div>
                          <div className="text-gray-400 leading-relaxed text-[13px] pl-3 border-l border-white/5" dangerouslySetInnerHTML={{ __html: round.highlighted_text_a }} />
                        </div>


                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500/80"></span>
                              <strong className="text-gray-200 text-sm font-medium">{result.persona_b}</strong>
                            </div>
                            <span className="text-xs font-mono text-gray-500">{round.b_score}</span>
                          </div>
                          <div className="text-gray-400 leading-relaxed text-[13px] pl-3 border-l border-white/5" dangerouslySetInnerHTML={{ __html: round.highlighted_text_b }} />
                        </div>
                      </div>
                    ))}


                    {loading && (result.agent_a_summary || result.agent_b_summary) && (
                      <div className="bg-card-bg rounded-xl border border-white/10 p-5 relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-full h-[1px] bg-white/20 animate-pulse"></div>
                        <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest mb-4 animate-pulse">Synthesizing...</div>

                        {result.agent_a_summary && (
                          <div className="mb-6">
                            <div className="flex items-center gap-2 mb-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-rose-500/80"></span>
                              <strong className="text-gray-200 text-sm font-medium">{result.persona_a || "Challenger"}</strong>
                            </div>
                            <div className="text-gray-400 leading-relaxed text-[13px] pl-3 border-l border-white/5"
                              dangerouslySetInnerHTML={{ __html: result.highlighted_text_a || result.agent_a_summary.replace(/\n/g, '<br/>') }} />
                          </div>
                        )}

                        {result.agent_b_summary && (
                          <div>
                            <div className="flex items-center gap-2 mb-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500/80"></span>
                              <strong className="text-gray-200 text-sm font-medium">{result.persona_b || "Supporter"}</strong>
                            </div>
                            <div className="text-gray-400 leading-relaxed text-[13px] pl-3 border-l border-white/5"
                              dangerouslySetInnerHTML={{ __html: result.highlighted_text_b || result.agent_b_summary.replace(/\n/g, '<br/>') }} />
                          </div>
                        )}
                      </div>
                    )}


                    {isPaused && (
                      <div className="bg-blue-900/10 rounded-xl border border-blue-500/20 p-5 no-print">
                        <h4 className="text-blue-400 text-sm font-medium mb-2">Human Intervention Required</h4>
                        <p className="text-gray-400 text-xs mb-4">Provide optional feedback or continue the state machine.</p>
                        <textarea
                          value={juryFeedback}
                          onChange={(e) => setJuryFeedback(e.target.value)}
                          placeholder="Optional guidance..."
                          className="w-full h-16 bg-black/20 border border-blue-500/20 rounded-lg p-2.5 text-xs text-gray-200 focus:outline-none focus:border-blue-500/40 mb-3 resize-none"
                        />
                        <div className="flex gap-3">
                          <button onClick={() => handleAnalyze('rewrite')} className="px-4 py-1.5 text-xs font-medium bg-transparent border border-gray-600 text-gray-300 rounded-lg hover:bg-white/5 transition-colors">
                            Continue Debate
                          </button>
                          <button onClick={() => handleAnalyze('mediator')} className="px-4 py-1.5 text-xs font-medium bg-white text-black rounded-lg hover:bg-gray-200 transition-colors">
                            Force Synthesis
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>


                <div className="space-y-6">
                  {result.final_summary && (
                    <div>
                      <h3 className="text-sm font-semibold text-gray-300 tracking-wide uppercase mb-4">Final Synthesis</h3>
                      <div className="bg-card-bg p-5 rounded-xl border border-white/5">
                        <div className="text-sm text-gray-300 leading-relaxed">
                          {result.final_summary}
                        </div>
                      </div>
                    </div>
                  )}

                  {result.synthesis_rouge && (
                    <div>
                      <h3 className="text-sm font-semibold text-gray-300 tracking-wide uppercase mb-4">Metrics</h3>
                      <div className="space-y-4">


                        <div className="grid grid-cols-2 gap-3">
                          <div className="bg-card-bg p-4 rounded-xl border border-white/5 flex flex-col justify-between h-20">
                            <span className="text-[10px] font-medium text-gray-500 uppercase tracking-widest">Rouge 1</span>
                            <span className="text-lg font-semibold text-gray-200">{result.synthesis_rouge.rouge1?.toFixed(3)}</span>
                          </div>
                          <div className="bg-card-bg p-4 rounded-xl border border-white/5 flex flex-col justify-between h-20">
                            <span className="text-[10px] font-medium text-gray-500 uppercase tracking-widest">Rouge L</span>
                            <span className="text-lg font-semibold text-gray-200">{result.synthesis_rouge.rougeL?.toFixed(3)}</span>
                          </div>
                        </div>


                        {result.synthesis_neutral && (
                          <div className="grid grid-cols-2 gap-3">
                            <div className="bg-card-bg p-4 rounded-xl border border-white/5 flex flex-col justify-between h-20">
                              <span className="text-[10px] font-medium text-gray-500 uppercase tracking-widest">Original Bias</span>
                              <span className="text-lg font-semibold text-gray-200">{result.synthesis_neutral.original_polarity?.toFixed(3)}</span>
                            </div>
                            <div className="bg-card-bg p-4 rounded-xl border border-white/5 flex flex-col justify-between h-20">
                              <span className="text-[10px] font-medium text-gray-500 uppercase tracking-widest">Synthesis Bias</span>
                              <span className="text-lg font-semibold text-gray-200">{result.synthesis_neutral.synthesis_polarity?.toFixed(3)}</span>
                            </div>
                          </div>
                        )}


                        {result.debate_influence && (
                          <div className="bg-card-bg p-5 rounded-xl border border-white/5">
                            <span className="block text-[10px] font-medium text-gray-500 uppercase tracking-widest mb-4">Debate Influence</span>
                            <div className="flex items-center gap-3">
                              <div className="flex-1">
                                <div className="flex justify-between text-xs mb-1.5">
                                  <span className="text-gray-400">Challenger</span>
                                  <span className="text-gray-200 font-mono">{result.debate_influence.challenger}%</span>
                                </div>
                                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                                  <div className="h-full bg-rose-500/50 rounded-full" style={{ width: `${result.debate_influence.challenger}%` }}></div>
                                </div>
                              </div>
                            </div>
                            <div className="flex items-center gap-3 mt-4">
                              <div className="flex-1">
                                <div className="flex justify-between text-xs mb-1.5">
                                  <span className="text-gray-400">Supporter</span>
                                  <span className="text-gray-200 font-mono">{result.debate_influence.supporter}%</span>
                                </div>
                                <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                                  <div className="h-full bg-emerald-500/50 rounded-full" style={{ width: `${result.debate_influence.supporter}%` }}></div>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}

                      </div>
                    </div>
                  )}
                </div>

              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;