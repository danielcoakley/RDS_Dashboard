import { getOrganizationReportEvidencePackage } from "../../../../lib/api";
import { readAppSession } from "../../../../lib/session";

type ReportEvidencePackageRouteContext = {
  params: Promise<{ reportId: string }>;
};

export async function GET(
  _request: Request,
  { params }: ReportEvidencePackageRouteContext
): Promise<Response> {
  const { reportId } = await params;
  const session = await readAppSession();

  if (!session.userId || !session.organizationId) {
    return new Response("Sign in to download this evidence package.", { status: 401 });
  }

  try {
    const exportFile = await getOrganizationReportEvidencePackage(
      session.userId,
      session.organizationId,
      reportId,
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
