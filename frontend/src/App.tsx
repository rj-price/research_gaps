import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Upload,
  Search,
  FileText,
  Loader2,
  CheckCircle2,
  AlertCircle,
  BookOpen,
  Download,
  Share2
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

const API_BASE_URL = 'http://localhost:8000';

interface AnalysisResult {
  report: string;
  summaries: string[];
}

interface TaskStatus {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  subject: string;
  result?: AnalysisResult;
  error?: string;
}

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [subject, setSubject] = useState('');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [taskStatus, setTaskStatus] = useState<TaskStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLLabelElement>) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files) {
      // Filter out only PDFs if needed, or just take array
      const droppedFiles = Array.from(e.dataTransfer.files).filter(
        file => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')
      );
      if (droppedFiles.length > 0) {
        setFiles(droppedFiles);
      }
    }
  };

  const startAnalysis = async () => {
    if (files.length === 0) return;
    setLoading(true);
    setTaskId(null);
    setTaskStatus(null);

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('subject', subject || 'the provided topics');

    try {
      const response = await axios.post(`${API_BASE_URL}/analyse`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setTaskId(response.data.task_id);
    } catch (err) {
      console.error(err);
      setTaskStatus({ status: 'failed', subject, error: 'Failed to start analysis.' });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let interval: number;

    if (taskId && (!taskStatus || (taskStatus.status !== 'completed' && taskStatus.status !== 'failed'))) {
      interval = window.setInterval(async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/status/${taskId}`);
          setTaskStatus(response.data);
          if (response.data.status === 'completed' || response.data.status === 'failed') {
            clearInterval(interval);
          }
        } catch (err) {
          console.error(err);
          clearInterval(interval);
        }
      }, 2000);
    }

    return () => clearInterval(interval);
  }, [taskId, taskStatus]);

  return (
    <div className="min-h-screen bg-background text-text selection:bg-primary/20 font-sans">
      {/* Header */}
      <header className="py-12 md:py-20 text-center px-4">
        <div className="max-w-3xl mx-auto flex flex-col items-center justify-center space-y-6">
          <div className="w-20 h-20 bg-white rounded-2xl shadow-sm flex items-center justify-center border border-primary/10">
            <BookOpen className="w-10 h-10 text-primary" />
          </div>
          <h1 className="text-4xl md:text-6xl font-serif font-bold text-text tracking-tight">
            Research Gap Identifier
          </h1>
          <p className="text-lg md:text-2xl text-text/80 max-w-2xl font-sans leading-relaxed">
            A professional academic tool to synthesise literature and identify novel research opportunities.
          </p>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-4 pb-24 space-y-12">
        {/* Upload & Configure Section */}
        <section className="bg-white p-8 md:p-12 rounded-3xl shadow-md border border-primary/10 transition-all duration-300 hover:shadow-lg">
          <div className="space-y-8">
            <div>
              <label className="block text-sm font-bold text-text mb-3 font-sans">
                Research Focus
              </label>
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-primary/50" />
                <input
                  type="text"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  placeholder="e.g., LLM Evaluation Methods"
                  className="w-full pl-12 pr-4 py-4 bg-background/50 border border-primary/20 rounded-xl focus:ring-2 focus:ring-primary/30 focus:border-primary transition-all outline-none text-text text-lg"
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-bold text-text mb-3 font-sans">
                Upload Literature (PDFs)
              </label>
              <label
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-200 ${isDragging
                  ? 'border-primary bg-primary/5'
                  : 'border-primary/30 bg-background/30 hover:bg-background/80 hover:border-primary/50'
                  }`}
              >
                <div className="flex flex-col items-center justify-center pt-5 pb-6 pointer-events-none">
                  <Upload className={`w-10 h-10 mb-4 ${isDragging ? 'text-primary' : 'text-primary/60'}`} />
                  <p className="mb-2 text-lg text-text/80 text-center px-4">
                    <span className="font-semibold text-primary">Click to upload</span> or drag and drop
                  </p>
                  <p className="text-sm text-text/60">PDF documents only</p>
                </div>
                <input type="file" className="hidden" multiple accept=".pdf" onChange={handleFileChange} />
              </label>
              {files.length > 0 && (
                <div className="mt-4 space-y-2">
                  {files.map((f, i) => (
                    <div key={i} className="text-sm flex items-center text-text bg-background border border-primary/10 p-3 rounded-lg shadow-sm">
                      <FileText className="w-4 h-4 mr-3 text-secondary" />
                      <span className="truncate flex-1 font-medium">{f.name}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="pt-6 flex justify-center">
              <button
                onClick={startAnalysis}
                disabled={loading || files.length === 0}
                className="w-full md:w-auto px-12 py-4 bg-cta hover:opacity-90 disabled:bg-cta/50 disabled:cursor-not-allowed hover:-translate-y-px text-white text-xl font-bold rounded-xl shadow-md hover:shadow-lg transition-all duration-200 flex items-center justify-center cursor-pointer"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-6 h-6 mr-3 animate-spin" />
                    Synthesising Literature...
                  </>
                ) : (
                  "Identify Research Gaps"
                )}
              </button>
            </div>
          </div>
        </section>

        {/* Task Status */}
        {taskStatus && (taskStatus.status === 'pending' || taskStatus.status === 'processing') && (
          <div className="bg-white p-12 rounded-3xl border border-primary/20 shadow-md animate-in fade-in slide-in-from-bottom-4 duration-500 text-center">
            <Loader2 className="w-12 h-12 mx-auto mb-6 text-primary animate-spin" />
            <h3 className="text-2xl font-serif font-bold text-text mb-3">Analysis in Progress</h3>
            <p className="text-text/70 text-lg">
              The multi-agent pipeline is currently synthesising your documents. This typically takes 30-60 seconds.
            </p>
          </div>
        )}

        {taskStatus && taskStatus.error && (
          <div className="bg-red-50 p-8 rounded-3xl border border-red-200 shadow-sm text-center">
            <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-500" />
            <h3 className="text-xl font-serif font-bold text-red-900 mb-2">Analysis Failed</h3>
            <p className="text-red-700">{taskStatus.error}</p>
          </div>
        )}

        {/* Results Section */}
        {taskStatus && taskStatus.status === 'completed' && taskStatus.result && (
          <section className="space-y-16 animate-in fade-in slide-in-from-bottom-8 duration-700 pt-8">
            <div className="bg-white p-8 md:p-14 rounded-3xl border border-primary/10 shadow-lg">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-10 pb-6 border-b border-primary/10">
                <h2 className="text-3xl md:text-4xl font-serif font-bold text-text mb-6 md:mb-0 flex items-center">
                  <CheckCircle2 className="w-8 h-8 mr-4 text-cta" />
                  Research Analysis Report
                </h2>
                <div className="flex space-x-3 w-full md:w-auto">
                  <button className="flex-1 md:flex-none justify-center items-center flex px-5 py-3 text-primary border border-primary/30 rounded-xl hover:bg-primary/5 hover:-translate-y-px transition-all cursor-pointer font-bold">
                    <Download className="w-5 h-5 mr-2" /> Export
                  </button>
                  <button className="flex-1 md:flex-none justify-center items-center flex px-5 py-3 text-primary border border-primary/30 rounded-xl hover:bg-primary/5 hover:-translate-y-px transition-all cursor-pointer font-bold">
                    <Share2 className="w-5 h-5 mr-2" /> Share
                  </button>
                </div>
              </div>

              <div className="prose prose-lg prose-slate prose-headings:font-serif prose-headings:text-text prose-a:text-primary hover:prose-a:text-secondary max-w-none text-text/90 leading-relaxed">
                <ReactMarkdown>{taskStatus.result.report}</ReactMarkdown>
              </div>
            </div>

            <div className="bg-background/50 p-8 md:p-12 rounded-3xl border border-primary/10">
              <h3 className="text-2xl md:text-3xl font-serif font-bold text-text mb-10 flex items-center justify-center text-center">
                <BookOpen className="w-8 h-8 mr-4 text-primary" />
                Source Material Summaries
              </h3>
              <div className="grid gap-8">
                {taskStatus.result.summaries.map((summary, idx) => (
                  <div key={idx} className="p-8 md:p-10 bg-white rounded-2xl shadow-sm border border-primary/5 transition-all hover:shadow-md">
                    <div className="prose prose-slate prose-headings:font-serif max-w-none text-text/80 text-lg leading-relaxed">
                      <ReactMarkdown>{summary}</ReactMarkdown>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="py-16 bg-white text-center border-t border-primary/10 mt-12">
        <div className="max-w-3xl mx-auto px-4">
          <p className="text-text/60 font-semibold text-lg mb-6">© 2026 Academic Research Intelligence Unit</p>
          <div className="flex flex-wrap justify-center gap-6">
            <a href="#" className="text-text/50 hover:text-primary transition-colours font-medium">Privacy Policy</a>
            <a href="#" className="text-text/50 hover:text-primary transition-colours font-medium">Terms of Service</a>
            <a href="#" className="text-text/50 hover:text-primary transition-colours font-medium">API Documentation</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
