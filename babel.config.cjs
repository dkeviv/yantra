module.exports = {
  presets: [
    'babel-preset-solid',
    ['@babel/preset-env', { targets: { node: 'current' } }],
    '@babel/preset-typescript',
  ],
};
