import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  CloudArrowUpIcon,
  DocumentIcon,
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';

export default function FileUpload({ 
  onFileSelect, 
  isUploading = false, 
  uploadProgress = 0,
  error = null,
  acceptedFileTypes = ['.pdf']
}) {
  const [uploadedFile, setUploadedFile] = useState(null);

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      let errorMsg = 'File upload failed';
      
      if (rejection.errors.find(e => e.code === 'file-too-large')) {
        errorMsg = 'File is too large. Maximum size is 10MB.';
      } else if (rejection.errors.find(e => e.code === 'file-invalid-type')) {
        errorMsg = `Invalid file type. Accepted types: ${acceptedFileTypes.join(', ')}`;
      }
      
      onFileSelect(null, errorMsg);
      return;
    }

    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setUploadedFile(file);
      onFileSelect(file, null);
    }
  }, [onFileSelect, acceptedFileTypes]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false,
  });

  const removeFile = () => {
    setUploadedFile(null);
    onFileSelect(null, null);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {!uploadedFile && !isUploading && (
        <div
          {...getRootProps()}
          className={clsx(
            'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200',
            isDragActive && !isDragReject
              ? 'border-primary-400 bg-primary-50 scale-105'
              : isDragReject
              ? 'border-red-400 bg-red-50'
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          )}
        >
          <input {...getInputProps()} />
          
          <div className="flex flex-col items-center">
            <CloudArrowUpIcon 
              className={clsx(
                'h-12 w-12 mb-4',
                isDragActive && !isDragReject 
                  ? 'text-primary-500' 
                  : isDragReject 
                  ? 'text-red-500' 
                  : 'text-gray-400'
              )} 
            />
            
            {isDragActive ? (
              isDragReject ? (
                <p className="text-red-600 font-medium">Invalid file type</p>
              ) : (
                <p className="text-primary-600 font-medium">Drop the file here</p>
              )
            ) : (
              <>
                <p className="text-lg font-medium text-gray-900 mb-2">
                  Drop your document here, or <span className="text-primary-600">browse</span>
                </p>
                <p className="text-sm text-gray-600 mb-4">
                  Supports: {acceptedFileTypes.join(', ')} • Max size: 10MB
                </p>
                <div className="bg-primary-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors">
                  Choose File
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* File preview */}
      {uploadedFile && !isUploading && (
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <DocumentIcon className="h-10 w-10 text-primary-600" />
              <div>
                <p className="font-medium text-gray-900">{uploadedFile.name}</p>
                <p className="text-sm text-gray-600">{formatFileSize(uploadedFile.size)}</p>
              </div>
            </div>
            <button
              onClick={removeFile}
              className="rounded-full p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 transition-colors"
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>
        </div>
      )}

      {/* Upload progress */}
      {isUploading && (
        <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
          <div className="flex items-center space-x-3 mb-4">
            <DocumentIcon className="h-10 w-10 text-primary-600" />
            <div className="flex-1">
              <p className="font-medium text-gray-900">{uploadedFile?.name}</p>
              <p className="text-sm text-gray-600">Uploading...</p>
            </div>
          </div>
          
          {/* Progress bar */}
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-primary-600 h-2 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <p className="text-sm text-gray-600 mt-2">{uploadProgress}% complete</p>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 mt-4">
          <div className="flex items-center">
            <ExclamationTriangleIcon className="h-5 w-5 text-red-500 mr-2" />
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* Success message */}
      {uploadedFile && !isUploading && !error && uploadProgress === 100 && (
        <div className="bg-green-50 border border-green-200 rounded-xl p-4 mt-4">
          <div className="flex items-center">
            <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
            <p className="text-green-700 text-sm">File uploaded successfully!</p>
          </div>
        </div>
      )}
    </div>
  );
}
