module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        'type-enum': ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'revert', 'ci', 'wip'],
        'scope-enum': ['vhdl', 'training', 'transform']
    }
}
