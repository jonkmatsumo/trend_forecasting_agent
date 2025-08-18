import { Injectable } from '@angular/core';
import { AbstractControl, ValidationErrors, ValidatorFn } from '@angular/forms';
import { ValidationError } from './error-handler.service';

export interface ValidationRule {
  validator: ValidatorFn;
  message: string;
  code?: string;
}

export interface FieldValidation {
  field: string;
  rules: ValidationRule[];
}

@Injectable({
  providedIn: 'root'
})
export class ValidationService {
  /**
   * Custom validators
   */
  static readonly validators = {
    /**
     * URL validator
     */
    url(): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        const urlPattern = /^https?:\/\/.+/;
        return urlPattern.test(control.value) ? null : { invalidUrl: true };
      };
    },

    /**
     * JSON validator
     */
    json(): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        try {
          JSON.parse(control.value);
          return null;
        } catch {
          return { invalidJson: true };
        }
      };
    },

    /**
     * API endpoint validator
     */
    apiEndpoint(): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        const endpointPattern = /^\/[a-zA-Z0-9\/\-_]+$/;
        return endpointPattern.test(control.value) ? null : { invalidEndpoint: true };
      };
    },

    /**
     * Keywords array validator
     */
    keywordsArray(): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        if (Array.isArray(control.value)) {
          const validKeywords = control.value.every(keyword => 
            typeof keyword === 'string' && keyword.trim().length > 0
          );
          return validKeywords ? null : { invalidKeywords: true };
        }
        
        return { invalidKeywords: true };
      };
    },

    /**
     * Timeframe validator
     */
    timeframe(): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        const validTimeframes = [
          'now 1-H', 'now 4-H', 'now 1-d', 'now 7-d',
          'today 1-m', 'today 3-m', 'today 12-m', 'today 5-y',
          'all'
        ];
        
        return validTimeframes.includes(control.value) ? null : { invalidTimeframe: true };
      };
    },

    /**
     * Model type validator
     */
    modelType(): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        const validTypes = ['prophet', 'arima', 'lstm', 'transformer'];
        return validTypes.includes(control.value) ? null : { invalidModelType: true };
      };
    },

    /**
     * Positive number validator
     */
    positiveNumber(): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        const num = Number(control.value);
        return !isNaN(num) && num > 0 ? null : { notPositiveNumber: true };
      };
    },

    /**
     * Range validator
     */
    range(min: number, max: number): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        const num = Number(control.value);
        if (isNaN(num)) return { notNumber: true };
        
        return num >= min && num <= max ? null : { outOfRange: { min, max, actual: num } };
      };
    },

    /**
     * No special characters validator
     */
    noSpecialChars(): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        const specialCharsPattern = /[<>:"\\|?*]/;
        return specialCharsPattern.test(control.value) ? { hasSpecialChars: true } : null;
      };
    },

    /**
     * File size validator
     */
    fileSize(maxSizeInMB: number): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        const file = control.value as File;
        if (!file || !file.size) return null;
        
        const maxSizeInBytes = maxSizeInMB * 1024 * 1024;
        return file.size <= maxSizeInBytes ? null : { fileTooLarge: { maxSize: maxSizeInMB, actual: file.size } };
      };
    },

    /**
     * File type validator
     */
    fileType(allowedTypes: string[]): ValidatorFn {
      return (control: AbstractControl): ValidationErrors | null => {
        if (!control.value) return null;
        
        const file = control.value as File;
        if (!file || !file.type) return null;
        
        return allowedTypes.includes(file.type) ? null : { invalidFileType: { allowed: allowedTypes, actual: file.type } };
      };
    }
  };

  /**
   * Default validation messages
   */
  static readonly messages = {
    required: 'This field is required',
    email: 'Please enter a valid email address',
    minlength: 'Minimum length is {0} characters',
    maxlength: 'Maximum length is {0} characters',
    pattern: 'Please enter a valid value',
    invalidUrl: 'Please enter a valid URL',
    invalidJson: 'Please enter valid JSON',
    invalidEndpoint: 'Please enter a valid API endpoint',
    invalidKeywords: 'Please enter valid keywords',
    invalidTimeframe: 'Please select a valid timeframe',
    invalidModelType: 'Please select a valid model type',
    notPositiveNumber: 'Please enter a positive number',
    notNumber: 'Please enter a valid number',
    outOfRange: 'Value must be between {0} and {1}',
    hasSpecialChars: 'Special characters are not allowed',
    fileTooLarge: 'File size must be less than {0}MB',
    invalidFileType: 'File type not supported'
  };

  /**
   * Get validation error message
   */
  getErrorMessage(control: AbstractControl, fieldName: string): string {
    if (!control.errors) return '';

    const errorKey = Object.keys(control.errors)[0];
    const error = control.errors[errorKey];
    
    let message = ValidationService.messages[errorKey as keyof typeof ValidationService.messages] || 'Invalid value';
    
    // Replace placeholders with actual values
    if (errorKey === 'minlength' || errorKey === 'maxlength') {
      message = message.replace('{0}', error.requiredLength);
    } else if (errorKey === 'outOfRange') {
      message = message.replace('{0}', error.min).replace('{1}', error.max);
    } else if (errorKey === 'fileTooLarge') {
      message = message.replace('{0}', error.maxSize);
    }
    
    return message;
  }

  /**
   * Get all validation errors for a form
   */
  getFormErrors(form: AbstractControl): ValidationError[] {
    const errors: ValidationError[] = [];
    
    if (form.errors) {
      Object.keys(form.errors).forEach(key => {
        errors.push({
          field: 'form',
          message: this.getErrorMessage(form, 'form'),
          code: key
        });
      });
    }
    
    if (form instanceof AbstractControl && 'controls' in form) {
      Object.keys((form as any).controls).forEach(controlName => {
        const control = form.get(controlName);
        if (control && control.errors) {
          Object.keys(control.errors).forEach(errorKey => {
            errors.push({
              field: controlName,
              message: this.getErrorMessage(control, controlName),
              code: errorKey
            });
          });
        }
      });
    }
    
    return errors;
  }

  /**
   * Validate a single field
   */
  validateField(control: AbstractControl, fieldName: string): ValidationError[] {
    const errors: ValidationError[] = [];
    
    if (control.errors) {
      Object.keys(control.errors).forEach(errorKey => {
        errors.push({
          field: fieldName,
          message: this.getErrorMessage(control, fieldName),
          code: errorKey
        });
      });
    }
    
    return errors;
  }

  /**
   * Check if form is valid and show errors if not
   */
  validateForm(form: AbstractControl): boolean {
    if (form.valid) return true;
    
    // Mark all fields as touched to show validation errors
    this.markFormGroupTouched(form);
    return false;
  }

  /**
   * Mark all form controls as touched
   */
  private markFormGroupTouched(formGroup: AbstractControl): void {
    if (formGroup instanceof AbstractControl && 'controls' in formGroup) {
      Object.keys((formGroup as any).controls).forEach(key => {
        const control = formGroup.get(key);
        if (control) {
          control.markAsTouched();
          if (control instanceof AbstractControl && 'controls' in control) {
            this.markFormGroupTouched(control);
          }
        }
      });
    }
  }

  /**
   * Create custom validation rule
   */
  createRule(validator: ValidatorFn, message: string, code?: string): ValidationRule {
    return { validator, message, code };
  }

  /**
   * Validate API request data
   */
  validateApiRequest(data: any): ValidationError[] {
    const errors: ValidationError[] = [];
    
    if (!data.url) {
      errors.push({ field: 'url', message: 'URL is required', code: 'required' });
    } else if (!data.url.startsWith('http')) {
      errors.push({ field: 'url', message: 'URL must start with http:// or https://', code: 'invalidUrl' });
    }
    
    if (!data.method) {
      errors.push({ field: 'method', message: 'HTTP method is required', code: 'required' });
    } else if (!['GET', 'POST', 'PUT', 'DELETE', 'PATCH'].includes(data.method.toUpperCase())) {
      errors.push({ field: 'method', message: 'Invalid HTTP method', code: 'invalidMethod' });
    }
    
    return errors;
  }

  /**
   * Validate agent request data
   */
  validateAgentRequest(data: any): ValidationError[] {
    const errors: ValidationError[] = [];
    
    if (!data.message || data.message.trim().length === 0) {
      errors.push({ field: 'message', message: 'Message is required', code: 'required' });
    } else if (data.message.length > 1000) {
      errors.push({ field: 'message', message: 'Message must be less than 1000 characters', code: 'maxlength' });
    }
    
    return errors;
  }
} 