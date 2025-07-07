/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';
import reactPlugin from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import prettierConfig from 'eslint-config-prettier';
import importPlugin from 'eslint-plugin-import';
import globals from 'globals';
import licenseHeader from 'eslint-plugin-license-header';
import noRelativeCrossPackageImports from './eslint-rules/no-relative-cross-package-imports.js';
import path from 'node:path'; // Use node: prefix for built-ins
import url from 'node:url';

// --- ESM way to get __dirname ---
const __filename = url.fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// --- ---

// Determine the monorepo root (assuming eslint.config.js is at the root)
const projectRoot = __dirname;

export default tseslint.config(
  {
    // Global ignores
    ignores: [
      'node_modules/*',
      'eslint.config.js',
      'packages/cli/dist/**',
      'packages/core/dist/**',
      'packages/server/dist/**',
      'eslint-rules/*',
      'bundle/**',
    ],
  },
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  reactHooks.configs['recommended-latest'],
  reactPlugin.configs.flat.recommended,
  reactPlugin.configs.flat['jsx-runtime'], // Add this if you are using React 17+
  {
    // Settings for eslint-plugin-react
    settings: {
      react: {
        version: 'detect',
      },
    },
  },
  {
    // Import specific config
    files: ['packages/cli/src/**/*.{ts,tsx}'], // Target only TS/TSX in the cli package
    plugins: {
      import: importPlugin,
    },
    settings: {
      'import/resolver': {
        node: true,
      },
    },
    rules: {
      ...importPlugin.configs.recommended.rules,
      ...importPlugin.configs.typescript.rules,
      'import/no-default-export': 'warn',
      'import/no-unresolved': 'off', // Disable for now, can be noisy with monorepos/paths
    },
  },
  {
    // General overrides and rules for the project (TS/TSX files)
    files: ['packages/*/src/**/*.{ts,tsx}'], // Target only TS/TSX in the cli package
    languageOptions: {
      globals: {
        ...globals.node,
        ...globals.es2021,
      },
    },
    rules: {
      // General Best Practice Rules (subset adapted for flat config)
      '@typescript-eslint/array-type': ['error', { default: 'array-simple' }],
      'arrow-body-style': ['error', 'as-needed'],
      curly: ['error', 'multi-line'],
      eqeqeq: ['error', 'always', { null: 'ignore' }],
      '@typescript-eslint/consistent-type-assertions': [
        'error',
        { assertionStyle: 'as' },
      ],
      '@typescript-eslint/explicit-member-accessibility': [
        'error',
        { accessibility: 'no-public' },
      ],
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-inferrable-types': [
        'error',
        { ignoreParameters: true, ignoreProperties: true },
      ],
      '@typescript-eslint/no-namespace': ['error', { allowDeclarations: true }],
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      'no-cond-assign': 'error',
      'no-debugger': 'error',
      'no-duplicate-case': 'error',
      'no-restricted-syntax': [
        'error',
        {
          selector: 'CallExpression[callee.name="require"]',
          message: 'Avoid using require(). Use ES6 imports instead.',
        },
        {
          selector: 'ThrowStatement > Literal:not([value=/^\\w+Error:/])',
          message:
            'Do not throw string literals or non-Error objects. Throw new Error("...") instead.',
        },
      ],
      'no-unsafe-finally': 'error',
      'no-unused-expressions': 'off', // Disable base rule
      '@typescript-eslint/no-unused-expressions': [
        // Enable TS version
        'error',
        { allowShortCircuit: true, allowTernary: true },
      ],
      'no-var': 'error',
      'object-shorthand': 'error',
      'one-var': ['error', 'never'],
      'prefer-arrow-callback': 'error',
      'prefer-const': ['error', { destructuring: 'all' }],
      radix: 'error',
      'default-case': 'error',
    },
  },
  {
    files: ['./**/*.{tsx,ts,js}'],
    ignores: ['packages/core/src/providers/**'],
    plugins: {
      'license-header': licenseHeader,
    },
    rules: {
      'license-header/header': [
        'error',
        [
          '/**',
          ' * @license',
          ' * Copyright 2025 Google LLC',
          ' * SPDX-License-Identifier: Apache-2.0',
          ' */',
        ],
      ],
    },
  },
  {
    files: ['packages/core/src/providers/**/*.{tsx,ts,js}'],
    plugins: {
      'license-header': licenseHeader,
    },
    rules: {
      'license-header/header': [
        'error',
        [
          '/**',
          ' * @license',
          ' * Copyright 2025 The Omni Code Project',
          ' * SPDX-License-Identifier: Apache-2.0',
          ' */',
        ],
      ],
    },
  },
  // extra settings for scripts that we run directly with node
  {
    files: ['./scripts/**/*.js', 'esbuild.config.js'],
    languageOptions: {
      globals: {
        ...globals.node,
        process: 'readonly',
        console: 'readonly',
      },
    },
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
    },
  },
  // Prettier config must be last
  prettierConfig,
  // extra settings for scripts that we run directly with node
  {
    files: ['./integration-tests/**/*.js'],
    languageOptions: {
      globals: {
        ...globals.node,
        process: 'readonly',
        console: 'readonly',
      },
    },
    rules: {
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
    },
  },
  // Custom eslint rules for this repo
  {
    files: ['packages/**/*.{js,jsx,ts,tsx}'],
    plugins: {
      custom: {
        rules: {
          'no-relative-cross-package-imports': noRelativeCrossPackageImports,
        },
      },
    },
    rules: {
      // Enable and configure your custom rule
      'custom/no-relative-cross-package-imports': [
        'error',
        {
          root: path.join(projectRoot, 'packages'),
        },
      ],
    },
  },
);
