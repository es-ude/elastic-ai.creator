module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        'type-enum': [2, "always", ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'revert', 'ci', 'wip']],
        'scope-enum':  [2, "always", ['all', 'vhdl', 'nn', 'transform', 'translation', 'templates', 'gh-workflow']]
    }
}
