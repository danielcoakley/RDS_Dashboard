import { getOrganizationReportEvidencePackageSnapshot } from "../../../../../../lib/api";
import { readAppSession } from "../../../../../../lib/session";

type ReportEvidencePackageDownloadRouteContext = {
  params: Promise<{ reportId: string; packageId: string }>;
};

export async function GET(
  _request: Request,
  { params }: ReportEvidencePackageDownloadRouteContext
): Promise<Response> {
  const { reportId, packageId } = await params;
  const session = await readAppSession();

  if (!session.userId || !session.organizationId) {
    return new Response("Sign in to download this evidence package.", { status: 401 });
  }

  try {
    const exportFile = await getOrganizationReportEvidencePackageSnapshot(
      session.userId,
      session.organizationId,
      reportId,
      packageId,
      session.authToken
    );
    return new Response(exportFile.content, {
      headers: {
        "Cache-Control": "no-store",
        "Content-Disposition": `attachment; filename="${exportFile.filename}"`,
        "Content-Type": exportFile.content_type
      }
    });
  } catch {
    return new Response("Report evidence package unavailable.", { status: 404 });
  }
}
