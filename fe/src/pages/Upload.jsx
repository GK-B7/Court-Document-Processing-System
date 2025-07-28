import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  DocumentTextIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  ArrowRightIcon,
} from '@heroicons/react/24/outline';
import FileUpload from '../components/FileUpload';
import LoadingSpinner from '../components/LoadingSpinner';
import axios from 'axios';
import toast from 'react-hot-toast';

export default function Upload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadError, setUploadError] = useState(null);
  const [uploadedJobId, setUploadedJobId] = useState(null);
  const navigate = useNavigate();

  const handleFileSelect = (file, error) => {
    setSelectedFile(file);
    setUploadError(error);
    setUploadedJobId(null);
    setUploadProgress(0);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadError(null);
    setUploadProgress(0);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post('/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          setUploadProgress(progress);
        },
      });

      setUploadedJobId(response.data.job_id);
      toast.success('Document uploaded successfully!');
      
      // Auto-redirect to job details after 2 seconds
      setTimeout(() => {
        navigate(`/jobs/${response.data.job_id}`);
      }, 2000);

    } catch (error) {
      console.error('Upload failed:', error);
      const errorMessage = error.response?.data?.detail || 'Upload failed. Please try again.';
      setUploadError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsUploading(false);
    }
  };

  const resetUpload = () => {
    setSelectedFile(null);
    setIsUploading(false);
    setUploadProgress(0);
    setUploadError(null);
    setUploadedJobId(null);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-fade-in">
      {/* Header */}
      <div className="text-center">
        <div className="flex items-center justify-center mb-4">
          <div className="p-3 bg-primary-100 rounded-full">
            <DocumentTextIcon className="h-8 w-8 text-primary-600" />
          </div>
        </div>
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Document</h1>
        <p className="text-lg text-gray-600">
          Upload banking documents to extract customer actions automatically
        </p>
      </div>

      {/* Information Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-blue-50 rounded-xl p-6 border border-blue-200">
          <div className="flex items-center mb-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <DocumentTextIcon className="h-5 w-5 text-blue-600" />
            </div>
            <h3 className="ml-3 font-semibold text-blue-900">Supported Files</h3>
          </div>
          <p className="text-blue-800 text-sm">
            PDF files up to 10MB. Banking documents with customer actions work best.
          </p>
        </div>

        <div className="bg-green-50 rounded-xl p-6 border border-green-200">
          <div className="flex items-center mb-3">
            <div className="p-2 bg-green-100 rounded-lg">
              <CheckCircleIcon className="h-5 w-5 text-green-600" />
            </div>
            <h3 className="ml-3 font-semibold text-green-900">What We Extract</h3>
          </div>
          <p className="text-green-800 text-sm">
            National IDs, customer information, and banking actions like freeze_funds or release_funds.
          </p>
        </div>

        <div className="bg-purple-50 rounded-xl p-6 border border-purple-200">
          <div className="flex items-center mb-3">
            <div className="p-2 bg-purple-100 rounded-lg">
              <ArrowRightIcon className="h-5 w-5 text-purple-600" />
            </div>
            <h3 className="ml-3 font-semibold text-purple-900">Processing Flow</h3>
          </div>
          <p className="text-purple-800 text-sm">
            Text extraction → Customer validation → Action matching → Review if needed → Execution.
          </p>
        </div>
      </div>

      {/* Upload Section */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8">
        {!uploadedJobId ? (
          <>
            <FileUpload
              onFileSelect={handleFileSelect}
              isUploading={isUploading}
              uploadProgress={uploadProgress}
              error={uploadError}
            />

            {/* Upload Button */}
            {selectedFile && !isUploading && !uploadError && (
              <div className="mt-6 text-center">
                <button
                  onClick={handleUpload}
                  className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 focus:ring-2 focus:ring-primary-500 transition-colors"
                >
                  <DocumentTextIcon className="h-5 w-5 mr-2" />
                  Process Document
                </button>
              </div>
            )}

            {/* Processing Status */}
            {isUploading && (
              <div className="mt-6 text-center">
                <LoadingSpinner size="lg" text="Uploading and processing..." />
              </div>
            )}
          </>
        ) : (
          /* Success State */
          <div className="text-center py-8">
            <div className="flex items-center justify-center mb-4">
              <div className="p-4 bg-green-100 rounded-full">
                <CheckCircleIcon className="h-12 w-12 text-green-600" />
              </div>
            </div>
            <h3 className="text-2xl font-bold text-gray-900 mb-2">Document Uploaded Successfully!</h3>
            <p className="text-gray-600 mb-6">
              Your document is being processed. You'll be redirected to the job details page shortly.
            </p>
            
            <div className="bg-gray-50 rounded-lg p-4 mb-6">
              <p className="text-sm text-gray-600 mb-1">Job ID</p>
              <p className="font-mono text-lg text-gray-900">{uploadedJobId}</p>
            </div>

            <div className="flex items-center justify-center space-x-4">
              <button
                onClick={() => navigate(`/jobs/${uploadedJobId}`)}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 transition-colors"
              >
                View Job Details
                <ArrowRightIcon className="h-4 w-4 ml-2" />
              </button>
              
              <button
                onClick={resetUpload}
                className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 transition-colors"
              >
                Upload Another
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Processing Information */}
      <div className="bg-gray-50 rounded-xl p-6">
        <div className="flex items-start">
          <InformationCircleIcon className="h-6 w-6 text-blue-500 mt-0.5 mr-3 flex-shrink-0" />
          <div>
            <h3 className="font-semibold text-gray-900 mb-2">How Document Processing Works</h3>
            <div className="text-sm text-gray-700 space-y-2">
              <div className="flex items-center">
                <div className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-bold mr-3">1</div>
                <span><strong>Text Extraction:</strong> Extract text from your uploaded document</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-bold mr-3">2</div>
                <span><strong>Pattern Matching:</strong> Find National IDs and associated actions</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-bold mr-3">3</div>
                <span><strong>Customer Validation:</strong> Verify customers exist in the database</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-bold mr-3">4</div>
                <span><strong>Action Matching:</strong> Match extracted actions to supported banking actions</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-bold mr-3">5</div>
                <span><strong>Review Routing:</strong> High-confidence items are auto-approved, others go to review</span>
              </div>
              <div className="flex items-center">
                <div className="w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-bold mr-3">6</div>
                <span><strong>Execution:</strong> Execute approved actions automatically</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
