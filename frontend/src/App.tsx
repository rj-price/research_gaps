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
  ChevronRight,
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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
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
      const response = await axios.post(`${API_BASE_URL}/analyze`, formData, {
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
    <div className="min-h-screen bg-academic-50 text-academic-900">
      {/* Header */}
      <header className="bg-white border-b border-academic-200 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <BookOpen className="w-8 h-8 text-academic-800" />
            <span className="text-xl font-bold tracking-tight text-academic-800 uppercase">Research Gap Identifier</span>
          </div>
          <div className="text-sm text-academic-500 font-medium">Academic Research Tool</div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column: Input */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white p-6 rounded-lg border border-academic-200 shadow-sm">
              <h2 className="text-lg font-semibold mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2" />
                Configure Analysis
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-academic-700 mb-1">Subject Matter</label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-academic-400" />
                    <input 
                      type="text" 
                      value={subject}
                      onChange={(e) => setSubject(e.target.value)}
                      placeholder="e.g., LLM Evaluation Methods"
                      className="w-full pl-10 pr-4 py-2 bg-academic-50 border border-academic-200 rounded-md focus:ring-2 focus:ring-academic-800 focus:border-transparent transition-all outline-none"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-academic-700 mb-1">Upload PDF Papers</label>
                  <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-academic-200 rounded-lg cursor-pointer bg-academic-50 hover:bg-academic-100 transition-colors">
                    <div className="flex flex-col items-center justify-center pt-5 pb-6">
                      <FileText className="w-10 h-10 mb-3 text-academic-400" />
                      <p className="mb-2 text-sm text-academic-500 font-medium">
                        <span className="font-semibold">Click to upload</span> or drag and drop
                      </p>
                      <p className="text-xs text-academic-400">PDF documents only</p>
                    </div>
                    <input type="file" className="hidden" multiple accept=".pdf" onChange={handleFileChange} />
                  </label>
                  {files.length > 0 && (
                    <div className="mt-2 space-y-1">
                      {files.map((f, i) => (
                        <div key={i} className="text-xs flex items-center text-academic-600 bg-white p-2 rounded border border-academic-100">
                          <FileText className="w-3 h-3 mr-2" />
                          <span className="truncate max-w-[200px]">{f.name}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <button 
                  onClick={startAnalysis}
                  disabled={loading || files.length === 0}
                  className="w-full py-2 bg-academic-800 hover:bg-academic-900 disabled:bg-academic-300 text-white font-semibold rounded-md shadow-sm transition-all flex items-center justify-center"
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                      Starting...
                    </>
                  ) : (
                    "Run Multi-Agent Analysis"
                  )}
                </button>
              </div>
            </div>

            {/* Task Status */}
            {taskStatus && (
              <div className="bg-white p-6 rounded-lg border border-academic-200 shadow-sm animate-in fade-in slide-in-from-bottom-2 duration-300">
                <h3 className="text-sm font-semibold uppercase tracking-wider text-academic-500 mb-4">Current Status</h3>
                <div className="space-y-4">
                  <div className="flex items-center">
                    {taskStatus.status === 'pending' || taskStatus.status === 'processing' ? (
                      <Loader2 className="w-5 h-5 mr-3 text-academic-800 animate-spin" />
                    ) : taskStatus.status === 'completed' ? (
                      <CheckCircle2 className="w-5 h-5 mr-3 text-green-600" />
                    ) : (
                      <AlertCircle className="w-5 h-5 mr-3 text-red-600" />
                    )}
                    <span className="font-medium capitalize">{taskStatus.status}</span>
                  </div>
                  {(taskStatus.status === 'pending' || taskStatus.status === 'processing') && (
                    <p className="text-sm text-academic-600">
                      The multi-agent pipeline is currently synthesizing your documents. This typically takes 30-60 seconds.
                    </p>
                  )}
                  {taskStatus.error && (
                    <p className="text-sm text-red-600 bg-red-50 p-3 rounded-md border border-red-100">
                      Error: {taskStatus.error}
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-2">
            {!taskStatus || (taskStatus.status !== 'completed' && taskStatus.status !== 'failed') ? (
              <div className="bg-white rounded-lg border border-academic-200 shadow-sm h-full min-h-[500px] flex flex-col items-center justify-center text-center p-8">
                <div className="w-20 h-20 bg-academic-50 rounded-full flex items-center justify-center mb-6">
                  <BookOpen className="w-10 h-10 text-academic-300" />
                </div>
                <h3 className="text-xl font-semibold text-academic-800 mb-2">Analysis Results</h3>
                <p className="text-academic-500 max-w-md">
                  Upload PDF research papers on the left and start the analysis to see synthesized research gaps and new study proposals here.
                </p>
              </div>
            ) : taskStatus.status === 'completed' && taskStatus.result ? (
              <div className="space-y-6">
                <div className="bg-white p-8 rounded-lg border border-academic-200 shadow-sm prose prose-slate max-w-none">
                  <div className="flex justify-between items-center mb-8 pb-4 border-b border-academic-100 not-prose">
                    <h2 className="text-2xl font-bold text-academic-900 m-0">Research Analysis Report</h2>
                    <div className="flex space-x-2">
                      <button className="p-2 text-academic-500 hover:text-academic-800 hover:bg-academic-50 rounded-md transition-colors">
                        <Download className="w-5 h-5" />
                      </button>
                      <button className="p-2 text-academic-500 hover:text-academic-800 hover:bg-academic-50 rounded-md transition-colors">
                        <Share2 className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                  <ReactMarkdown>{taskStatus.result.report}</ReactMarkdown>
                </div>

                <div className="bg-white p-8 rounded-lg border border-academic-200 shadow-sm">
                  <h3 className="text-lg font-bold text-academic-900 mb-6 flex items-center">
                    <FileText className="w-5 h-5 mr-2" />
                    Source Material Summaries
                  </h3>
                  <div className="space-y-6">
                    {taskStatus.result.summaries.map((summary, idx) => (
                      <div key={idx} className="p-6 bg-academic-50 rounded-lg border border-academic-100 prose prose-sm max-w-none">
                        <ReactMarkdown>{summary}</ReactMarkdown>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 py-8 bg-academic-800 text-academic-300 text-center text-sm border-t border-academic-900">
        <div className="max-w-6xl mx-auto px-4">
          <p>© 2026 Academic Research Intelligence Unit</p>
          <div className="flex justify-center space-x-6 mt-4">
            <a href="#" className="hover:text-white transition-colors">Privacy Policy</a>
            <a href="#" className="hover:text-white transition-colors">Terms of Service</a>
            <a href="#" className="hover:text-white transition-colors">API Documentation</a>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
