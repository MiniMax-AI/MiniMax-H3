#!/usr/bin/env node
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

const skillDir = path.resolve(__dirname, '..', 'skills', 'h3-prompt-writing');

function printHelp() {
  console.log('MiniMax H3 prompt writing skill');
  console.log('');
  console.log('Commands:');
  console.log('  install [--dir PATH] [--force]  Install skill into ~/.codex/skills');
  console.log('');
  console.log('Running without a command installs the skill.');
  console.log('  --help                          Show this help');
}

function install(options) {
  if (!fs.existsSync(skillDir)) {
    console.error('Skill directory not found in this package.');
    process.exit(1);
  }

  const codexHome = process.env.CODEX_HOME || path.join(os.homedir(), '.codex');
  const root = options.dir ? path.resolve(options.dir) : path.join(codexHome, 'skills');
  const target = path.join(root, 'h3-prompt-writing');

  if (fs.existsSync(target) && !options.force) {
    console.error(`Skill already exists at ${target}`);
    console.error('Use --force to replace it.');
    process.exit(1);
  }

  fs.mkdirSync(root, { recursive: true });
  fs.cpSync(skillDir, target, { recursive: true });
  console.log(`Installed MiniMax H3 prompt writing skill to ${target}`);
}

function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'install';
  const options = { dir: null, force: false };

  for (let i = 1; i < args.length; i += 1) {
    if (args[i] === '--dir') {
      options.dir = args[i + 1];
      i += 1;
    } else if (args[i] === '--force') {
      options.force = true;
    }
  }

  if (command === 'install') {
    install(options);
  } else {
    printHelp();
  }
}

main();
