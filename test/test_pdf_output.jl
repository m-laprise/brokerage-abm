using Test

include(joinpath(@__DIR__, "..", "scripts", "paper", "pdf_output.jl"))

"""Construct PDF byte fixtures without running TeX or touching publication outputs."""
function pdf_fixture(;
    date="20260910140338",
    id="<0123456789abcdef>",
    content="BT /F1 12 Tf (Output 4.730) Tj ET",
    title="Report",
)
    body = "%PDF-1.7\n1 0 obj\n<< /Length $(sizeof(content)) >>\nstream\n" *
        "$content\nendstream\nendobj\n2 0 obj\n<<\n/Title ($title)\n" *
        "/CreationDate (D:$date-04'00')\n/ModDate (D:$date-04'00')\n>>\nendobj\n"
    tail = "xref\n0 1\n0000000000 65535 f\ntrailer\n" *
        "<< /Info 2 0 R /ID [$id $id] >>\nstartxref\n$(sizeof(body))\n%%EOF\n"
    return collect(codeunits(body * tail))
end

@testset "PDF output stability" begin
    original = pdf_fixture()
    metadata_only = pdf_fixture(; date="20260914120338", id="<fedcba9876543210>")
    @test pdf_comparison_bytes(original) == pdf_comparison_bytes(metadata_only)
    @test pdf_comparison_bytes(pdf_fixture(; id="(identifier one)")) ==
        pdf_comparison_bytes(pdf_fixture(; id="(identifier two)"))
    @test pdf_comparison_bytes(pdf_fixture(; id="(escaped\\(id\\))")) ==
        pdf_comparison_bytes(pdf_fixture(; id="(other\\(id\\))"))
    changed = [
        pdf_fixture(; content="BT /F1 12 Tf (Output 4.731) Tj ET"),
        pdf_fixture(; content="BT /F2 12 Tf (Output 4.730) Tj ET"),
        pdf_fixture(; title="Edited"),
        pdf_fixture(; content="/CreationDate (D:20260910140338-04'00')"),
        pdf_fixture(; content="/ID [<0123456789abcdef> <0123456789abcdef>]"),
    ]
    @test all(
        pdf_comparison_bytes(value) != pdf_comparison_bytes(original) for value in changed
    )
    date_text = [
        pdf_fixture(; content="/CreationDate (D:$date-04'00')") for
        date in ("20260910140338", "20260914120338")
    ]
    id_text = [
        pdf_fixture(; content="/ID [<$id> <$id>]") for
        id in ("0123456789abcdef", "fedcba9876543210")
    ]
    @test pdf_comparison_bytes(date_text[1]) != pdf_comparison_bytes(date_text[2])
    @test pdf_comparison_bytes(id_text[1]) != pdf_comparison_bytes(id_text[2])
    @test pdf_comparison_bytes(UInt8[0xff, 0x00]) == UInt8[0xff, 0x00]
    @test pdf_comparison_bytes(
        pdf_fixture(; content=String(UInt8[0xff, 0x00]))
    ) isa Vector{UInt8}

    mktempdir() do directory
        source = joinpath(directory, "built.pdf")
        destination = joinpath(directory, "published.pdf")
        write(source, original)
        @test update_pdf(source, destination)
        @test read(destination) == original
        @test !update_pdf(source, destination)
        stamp = mtime(destination)
        write(source, metadata_only)
        @test !update_pdf(source, destination)
        @test read(destination) == original && mtime(destination) == stamp
        write(source, first(changed))
        @test update_pdf(source, destination)
        @test read(destination) == first(changed)
        write(source, "not a PDF")
        @test_throws ErrorException update_pdf(source, destination)
        @test read(destination) == first(changed)
        @test_throws SystemError update_pdf(joinpath(directory, "missing.pdf"), destination)
    end
end
