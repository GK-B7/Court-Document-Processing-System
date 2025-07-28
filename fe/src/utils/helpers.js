import { format, formatDistanceToNow, parseISO, isValid } from 'date-fns';

// ──────────────────────────────────────────────────────────────────────────────
// Date & Time Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Format a date string or Date object to a readable format
 * @param {string|Date} date - Date to format
 * @param {string} formatStr - Format string (default: 'MMM dd, yyyy HH:mm')
 * @returns {string} Formatted date string
 */
export const formatDate = (date, formatStr = 'MMM dd, yyyy HH:mm') => {
  if (!date) return 'Unknown';
  
  try {
    const dateObj = typeof date === 'string' ? parseISO(date) : date;
    if (!isValid(dateObj)) return 'Invalid Date';
    return format(dateObj, formatStr);
  } catch (error) {
    console.error('Date formatting error:', error);
    return 'Invalid Date';
  }
};

/**
 * Format a date as relative time (e.g., "2 hours ago")
 * @param {string|Date} date - Date to format
 * @returns {string} Relative time string
 */
export const formatRelativeTime = (date) => {
  if (!date) return 'Unknown';
  
  try {
    const dateObj = typeof date === 'string' ? parseISO(date) : date;
    if (!isValid(dateObj)) return 'Invalid Date';
    return formatDistanceToNow(dateObj, { addSuffix: true });
  } catch (error) {
    console.error('Relative time formatting error:', error);
    return 'Unknown';
  }
};

/**
 * Calculate duration between two dates
 * @param {string|Date} startDate - Start date
 * @param {string|Date} endDate - End date (default: now)
 * @returns {string} Duration string
 */
export const calculateDuration = (startDate, endDate = new Date()) => {
  if (!startDate) return 'Unknown';
  
  try {
    const start = typeof startDate === 'string' ? parseISO(startDate) : startDate;
    const end = typeof endDate === 'string' ? parseISO(endDate) : endDate;
    
    if (!isValid(start) || !isValid(end)) return 'Invalid';
    
    const diffMs = Math.abs(end - start);
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffDays > 0) return `${diffDays}d ${diffHours % 24}h`;
    if (diffHours > 0) return `${diffHours}h ${diffMins % 60}m`;
    if (diffMins > 0) return `${diffMins}m ${diffSecs % 60}s`;
    return `${diffSecs}s`;
  } catch (error) {
    console.error('Duration calculation error:', error);
    return 'Unknown';
  }
};

// ──────────────────────────────────────────────────────────────────────────────
// String Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Truncate text to a specified length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length (default: 100)
 * @param {string} suffix - Suffix to add (default: '...')
 * @returns {string} Truncated text
 */
export const truncateText = (text, maxLength = 100, suffix = '...') => {
  if (!text || typeof text !== 'string') return '';
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength - suffix.length) + suffix;
};

/**
 * Convert string to title case
 * @param {string} str - String to convert
 * @returns {string} Title case string
 */
export const toTitleCase = (str) => {
  if (!str || typeof str !== 'string') return '';
  return str.replace(/\w\S*/g, (txt) => 
    txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase()
  );
};

/**
 * Convert snake_case or kebab-case to readable format
 * @param {string} str - String to convert
 * @returns {string} Readable string
 */
export const toReadableString = (str) => {
  if (!str || typeof str !== 'string') return '';
  return str
    .replace(/[_-]/g, ' ')
    .replace(/\b\w/g, char => char.toUpperCase());
};

/**
 * Generate initials from a name
 * @param {string} name - Full name
 * @param {number} maxInitials - Maximum number of initials (default: 2)
 * @returns {string} Initials
 */
export const getInitials = (name, maxInitials = 2) => {
  if (!name || typeof name !== 'string') return '';
  
  return name
    .split(' ')
    .slice(0, maxInitials)
    .map(word => word.charAt(0).toUpperCase())
    .join('');
};

// ──────────────────────────────────────────────────────────────────────────────
// Number & File Size Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Format file size in bytes to human readable format
 * @param {number} bytes - File size in bytes
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} Formatted file size
 */
export const formatFileSize = (bytes, decimals = 2) => {
  if (!bytes || bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

/**
 * Format number with commas for thousands
 * @param {number} num - Number to format
 * @returns {string} Formatted number
 */
export const formatNumber = (num) => {
  if (num == null || isNaN(num)) return '0';
  return num.toLocaleString();
};

/**
 * Format percentage
 * @param {number} value - Value to convert to percentage
 * @param {number} total - Total value (default: 1 for already calculated percentages)
 * @param {number} decimals - Decimal places (default: 1)
 * @returns {string} Formatted percentage
 */
export const formatPercentage = (value, total = 1, decimals = 1) => {
  if (value == null || total == null || total === 0) return '0%';
  const percentage = (value / total) * 100;
  return `${percentage.toFixed(decimals)}%`;
};

// ──────────────────────────────────────────────────────────────────────────────
// Job Status Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Get status color for UI components
 * @param {string} status - Job status
 * @returns {string} Color class name
 */
export const getStatusColor = (status) => {
  const statusColors = {
    pending: 'gray',
    processing: 'blue',
    completed: 'green',
    review_required: 'yellow',
    failed: 'red',
    cancelled: 'gray',
  };
  
  return statusColors[status] || 'gray';
};

/**
 * Get priority color for UI components
 * @param {string|number} priority - Priority level
 * @returns {string} Color class name
 */
export const getPriorityColor = (priority) => {
  const priorityColors = {
    critical: 'red',
    high: 'orange',
    medium: 'yellow',
    low: 'green',
    1: 'red',     // Critical
    2: 'orange',  // High
    3: 'yellow',  // Medium
    4: 'green',   // Low
    5: 'green',   // Low
  };
  
  return priorityColors[priority] || 'gray';
};

/**
 * Convert numeric priority to string
 * @param {number} priority - Numeric priority (1-5)
 * @returns {string} Priority string
 */
export const getPriorityString = (priority) => {
  const priorityMap = {
    1: 'critical',
    2: 'high',
    3: 'medium',
    4: 'low',
    5: 'low'
  };
  
  return priorityMap[priority] || 'medium';
};

/**
 * Check if a job is in progress
 * @param {string} status - Job status
 * @returns {boolean} True if job is in progress
 */
export const isJobInProgress = (status) => {
  return ['pending', 'processing'].includes(status);
};

/**
 * Check if a job is completed (successfully or failed)
 * @param {string} status - Job status
 * @returns {boolean} True if job is completed
 */
export const isJobCompleted = (status) => {
  return ['completed', 'failed', 'cancelled'].includes(status);
};

// ──────────────────────────────────────────────────────────────────────────────
// File & Validation Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Get file extension from filename
 * @param {string} filename - File name
 * @returns {string} File extension (lowercase)
 */
export const getFileExtension = (filename) => {
  if (!filename || typeof filename !== 'string') return '';
  return filename.split('.').pop()?.toLowerCase() || '';
};

/**
 * Check if file type is supported
 * @param {string} filename - File name
 * @param {string[]} allowedTypes - Allowed file extensions (default: PDF, DOC, DOCX, TXT)
 * @returns {boolean} True if file type is supported
 */
export const isFileTypeSupported = (filename, allowedTypes = ['.pdf', '.doc', '.docx', '.txt']) => {
  const extension = '.' + getFileExtension(filename);
  return allowedTypes.map(type => type.toLowerCase()).includes(extension.toLowerCase());
};

/**
 * Validate email format
 * @param {string} email - Email address
 * @returns {boolean} True if email is valid
 */
export const isValidEmail = (email) => {
  if (!email || typeof email !== 'string') return false;
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

/**
 * Validate national ID format (basic validation)
 * @param {string} nationalId - National ID
 * @returns {boolean} True if format appears valid
 */
export const isValidNationalId = (nationalId) => {
  if (!nationalId || typeof nationalId !== 'string') return false;
  // Basic validation: 10 digits
  const idRegex = /^\d{10}$/;
  return idRegex.test(nationalId.replace(/\s/g, ''));
};

// ──────────────────────────────────────────────────────────────────────────────
// URL & Navigation Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Build query string from object
 * @param {Object} params - Parameters object
 * @returns {string} Query string
 */
export const buildQueryString = (params) => {
  if (!params || typeof params !== 'object') return '';
  
  const searchParams = new URLSearchParams();
  
  Object.entries(params).forEach(([key, value]) => {
    if (value !== null && value !== undefined && value !== '') {
      searchParams.append(key, String(value));
    }
  });
  
  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : '';
};

/**
 * Parse query string to object
 * @param {string} queryString - Query string (with or without ?)
 * @returns {Object} Parameters object
 */
export const parseQueryString = (queryString) => {
  if (!queryString || typeof queryString !== 'string') return {};
  
  const cleanQuery = queryString.startsWith('?') ? queryString.slice(1) : queryString;
  const searchParams = new URLSearchParams(cleanQuery);
  const params = {};
  
  for (const [key, value] of searchParams) {
    params[key] = value;
  }
  
  return params;
};

// ──────────────────────────────────────────────────────────────────────────────
// Storage Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Safely get item from localStorage
 * @param {string} key - Storage key
 * @param {any} defaultValue - Default value if key doesn't exist
 * @returns {any} Stored value or default
 */
export const getFromStorage = (key, defaultValue = null) => {
  try {
    const item = localStorage.getItem(key);
    return item ? JSON.parse(item) : defaultValue;
  } catch (error) {
    console.error('LocalStorage get error:', error);
    return defaultValue;
  }
};

/**
 * Safely set item in localStorage
 * @param {string} key - Storage key
 * @param {any} value - Value to store
 * @returns {boolean} True if successful
 */
export const setToStorage = (key, value) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.error('LocalStorage set error:', error);
    return false;
  }
};

/**
 * Remove item from localStorage
 * @param {string} key - Storage key
 * @returns {boolean} True if successful
 */
export const removeFromStorage = (key) => {
  try {
    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error('LocalStorage remove error:', error);
    return false;
  }
};

// ──────────────────────────────────────────────────────────────────────────────
// Array & Object Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Safely access nested object properties
 * @param {Object} obj - Object to access
 * @param {string} path - Dot notation path (e.g., 'user.profile.name')
 * @param {any} defaultValue - Default value if path doesn't exist
 * @returns {any} Value at path or default
 */
export const getNestedValue = (obj, path, defaultValue = null) => {
  if (!obj || !path) return defaultValue;
  
  try {
    return path.split('.').reduce((current, key) => {
      return current && current[key] !== undefined ? current[key] : undefined;
    }, obj) ?? defaultValue;
  } catch (error) {
    return defaultValue;
  }
};

/**
 * Remove duplicates from array based on a key
 * @param {Array} array - Array to deduplicate
 * @param {string} key - Key to compare (optional, compares whole objects if not provided)
 * @returns {Array} Deduplicated array
 */
export const removeDuplicates = (array, key = null) => {
  if (!Array.isArray(array)) return [];
  
  if (key) {
    const seen = new Set();
    return array.filter(item => {
      const value = getNestedValue(item, key);
      if (seen.has(value)) return false;
      seen.add(value);
      return true;
    });
  }
  
  return [...new Set(array)];
};

/**
 * Group array of objects by a key
 * @param {Array} array - Array to group
 * @param {string} key - Key to group by
 * @returns {Object} Grouped object
 */
export const groupBy = (array, key) => {
  if (!Array.isArray(array)) return {};
  
  return array.reduce((groups, item) => {
    const value = getNestedValue(item, key);
    const groupKey = value ?? 'undefined';
    
    if (!groups[groupKey]) {
      groups[groupKey] = [];
    }
    
    groups[groupKey].push(item);
    return groups;
  }, {});
};

// ──────────────────────────────────────────────────────────────────────────────
// Async Utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Sleep for specified milliseconds
 * @param {number} ms - Milliseconds to sleep
 * @returns {Promise} Promise that resolves after delay
 */
export const sleep = (ms) => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

/**
 * Debounce function calls
 * @param {Function} func - Function to debounce
 * @param {number} delay - Delay in milliseconds
 * @returns {Function} Debounced function
 */
export const debounce = (func, delay) => {
  let timeoutId;
  
  return function (...args) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func.apply(this, args), delay);
  };
};

/**
 * Throttle function calls
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in milliseconds
 * @returns {Function} Throttled function
 */
export const throttle = (func, limit) => {
  let inThrottle;
  
  return function (...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

// ──────────────────────────────────────────────────────────────────────────────
// Constants and Defaults
// ──────────────────────────────────────────────────────────────────────────────

export const DEFAULT_DATE_FORMAT = 'MMM dd, yyyy HH:mm';
export const DEFAULT_SHORT_DATE_FORMAT = 'MMM dd, yyyy';
export const DEFAULT_TIME_FORMAT = 'HH:mm';

export const FILE_SIZE_LIMITS = {
  MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
  SUPPORTED_TYPES: ['.pdf', '.doc', '.docx', '.txt'],
};

export const JOB_STATUSES = {
  PENDING: 'pending',
  PROCESSING: 'processing',
  COMPLETED: 'completed',
  REVIEW_REQUIRED: 'review_required',
  FAILED: 'failed',
  CANCELLED: 'cancelled',
};

export const PRIORITY_LEVELS = {
  CRITICAL: 'critical',
  HIGH: 'high',
  MEDIUM: 'medium',
  LOW: 'low',
};

// Export all utilities as default
export default {
  // Date utilities
  formatDate,
  formatRelativeTime,
  calculateDuration,
  
  // String utilities
  truncateText,
  toTitleCase,
  toReadableString,
  getInitials,
  
  // Number utilities
  formatFileSize,
  formatNumber,
  formatPercentage,
  
  // Status utilities
  getStatusColor,
  getPriorityColor,
  getPriorityString,
  isJobInProgress,
  isJobCompleted,
  
  // File utilities
  getFileExtension,
  isFileTypeSupported,
  isValidEmail,
  isValidNationalId,
  
  // URL utilities
  buildQueryString,
  parseQueryString,
  
  // Storage utilities
  getFromStorage,
  setToStorage,
  removeFromStorage,
  
  // Object utilities
  getNestedValue,
  removeDuplicates,
  groupBy,
  
  // Async utilities
  sleep,
  debounce,
  throttle,
  
  // Constants
  DEFAULT_DATE_FORMAT,
  DEFAULT_SHORT_DATE_FORMAT,
  DEFAULT_TIME_FORMAT,
  FILE_SIZE_LIMITS,
  JOB_STATUSES,
  PRIORITY_LEVELS,
};
