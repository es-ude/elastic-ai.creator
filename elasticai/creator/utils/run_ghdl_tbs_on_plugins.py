import click
from .hw_test_runner import TestBenchReport, VhdlTestBenchRunner
import xml.etree.ElementTree as ET
import logging


@click.command()
@click.option(
    "-o",
    "--output-file",
    default="test_bench_results.xml",
    help="the file that the xml output will be written to, default: test_bench_results.xml",
)
@click.option(
    "-v", "--verbose", help="wether to print debug output", is_flag=True, default=False
)
def main(verbose, output_file):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    report = TestBenchReport(ET.Element("testsuites"))
    test_benches = VhdlTestBenchRunner(report)
    test_benches.run()
    report.dump(output_file)
