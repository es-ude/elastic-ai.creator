module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        'type-enum': [2, "always", ['feat', 'fix', 'docs', 'refactor', 'revert', 'chore', 'wip']],
        'scope-enum': [2, "always", ['template', 'translation', 'nn', 'transformation', 'unit', 'integration']]
    }
}
