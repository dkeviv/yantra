/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'jsdom',
  setupFilesAfterEnv: ['<rootDir>/jest.setup.cjs'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src-ui/$1',
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
  },
  transform: {
    '^.+\\.(t|j)sx?$': ['babel-jest', { 
      presets: [
        'babel-preset-solid',
        ['@babel/preset-env', { targets: { node: 'current' } }],
        '@babel/preset-typescript',
      ],
    }],
  },
  testMatch: [
    '<rootDir>/src-ui/components/__tests__/**/*.test.{ts,tsx}',
  ],
  collectCoverageFrom: [
    'src-ui/components/**/*.{ts,tsx}',
    '!src-ui/components/**/*.test.{ts,tsx}',
    '!src-ui/components/__tests__/**',
  ],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx'],
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
  globals: {
    'ts-jest': {
      tsconfig: {
        jsx: 'preserve',
      },
    },
  },
};
