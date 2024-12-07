module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        'type-enum': [2, "always", ['feat', 'fix', 'docs', 'style', 'refactor', 'revert', 'chore', 'wip', 'perf', 'build', 'test', 'style']],
        'header-max-length': [1, 'always', 100]
    }
}
