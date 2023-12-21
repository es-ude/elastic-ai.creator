module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        'type-enum': [2, "always", ['feat', 'fix', 'docs', 'style', 'refactor', 'revert', 'chore', 'wip', 'perf']],
        'header-max-length': [1, 'always', 100]
    }
}
